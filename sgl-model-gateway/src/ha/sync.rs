//! HA state synchronization module
//!
//! Handles synchronization of worker and policy states across HA cluster nodes

use std::sync::Arc;

use tracing::debug;

use super::{
    crdt::SKey,
    gossip::NodeStatus,
    stores::{
        tree_state_key, PolicyState, RateLimitConfig, StateStores, WorkerState,
        GLOBAL_RATE_LIMIT_COUNTER_KEY, GLOBAL_RATE_LIMIT_KEY,
    },
    tree_ops::{TreeOperation, TreeState},
};

/// HA sync manager for coordinating state synchronization
#[derive(Clone, Debug)]
pub struct HASyncManager {
    stores: Arc<StateStores>,
    self_name: String,
}

impl HASyncManager {
    pub fn new(stores: Arc<StateStores>, self_name: String) -> Self {
        Self { stores, self_name }
    }

    /// Get the node name (actor) for this sync manager
    pub fn self_name(&self) -> &str {
        &self.self_name
    }

    /// Sync worker state to HA stores
    pub fn sync_worker_state(
        &self,
        worker_id: String,
        model_id: String,
        url: String,
        health: bool,
        load: f64,
    ) {
        let key = SKey::new(worker_id.clone());

        // Get current version if exists, otherwise start at 1
        let current_version = self
            .stores
            .worker
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);
        let new_version = current_version + 1;

        let state = WorkerState {
            worker_id: worker_id.clone(),
            model_id,
            url,
            health,
            load,
            version: new_version,
        };

        // Use self node name as actor
        let actor = self.self_name.clone();
        self.stores.worker.insert(key, state, actor);
        debug!(
            "Synced worker state to HA: {} (version: {})",
            worker_id, new_version
        );
    }

    /// Remove worker state from HA stores
    pub fn remove_worker_state(&self, worker_id: &str) {
        let key = SKey::new(worker_id.to_string());
        self.stores.worker.remove(&key);
        debug!("Removed worker state from HA: {}", worker_id);
    }

    /// Sync policy state to HA stores
    pub fn sync_policy_state(&self, model_id: String, policy_type: String, config: Vec<u8>) {
        let key = SKey::new(format!("policy:{}", model_id));

        // Get current version if exists, otherwise start at 1
        let current_version = self
            .stores
            .policy
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);
        let new_version = current_version + 1;

        let state = PolicyState {
            model_id: model_id.clone(),
            policy_type,
            config,
            version: new_version,
        };

        // Use self node name as actor
        let actor = self.self_name.clone();
        self.stores.policy.insert(key, state, actor);
        debug!(
            "Synced policy state to HA: model={} (version: {})",
            model_id, new_version
        );
    }

    /// Remove policy state from HA stores
    pub fn remove_policy_state(&self, model_id: &str) {
        let key = SKey::new(format!("policy:{}", model_id));
        self.stores.policy.remove(&key);
        debug!("Removed policy state from HA: model={}", model_id);
    }

    /// Get worker state from HA stores
    pub fn get_worker_state(&self, worker_id: &str) -> Option<WorkerState> {
        let key = SKey::new(worker_id.to_string());
        self.stores.worker.get(&key)
    }

    /// Get all worker states from HA stores
    pub fn get_all_worker_states(&self) -> Vec<WorkerState> {
        self.stores.worker.all().into_values().collect()
    }

    /// Get policy state from HA stores
    pub fn get_policy_state(&self, model_id: &str) -> Option<PolicyState> {
        let key = SKey::new(format!("policy:{}", model_id));
        self.stores.policy.get(&key)
    }

    /// Get all policy states from HA stores
    pub fn get_all_policy_states(&self) -> Vec<PolicyState> {
        self.stores.policy.all().into_values().collect()
    }

    /// Apply worker state update from remote node
    /// The actor should be extracted from the state update context (e.g., from StateUpdate message)
    pub fn apply_remote_worker_state(&self, state: WorkerState, actor: Option<String>) {
        let key = SKey::new(state.worker_id.clone());
        // Use provided actor, or fallback to a default if not available
        // In practice, actor should come from the StateUpdate message
        let actor = actor.unwrap_or_else(|| "remote".to_string());

        // Check if we should update based on version
        let current_version = self
            .stores
            .worker
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);

        if state.version > current_version {
            self.stores.worker.insert(key, state.clone(), actor.clone());
            debug!(
                "Applied remote worker state update: {} (version: {} -> {})",
                state.worker_id, current_version, state.version
            );
        } else {
            debug!(
                "Skipped remote worker state update: {} (version {} <= current {})",
                state.worker_id, state.version, current_version
            );
        }
    }

    /// Apply policy state update from remote node
    /// The actor should be extracted from the state update context (e.g., from StateUpdate message)
    pub fn apply_remote_policy_state(&self, state: PolicyState, actor: Option<String>) {
        let key = SKey::new(format!("policy:{}", state.model_id));
        // Use provided actor, or fallback to a default if not available
        let actor = actor.unwrap_or_else(|| "remote".to_string());

        // Check if we should update based on version
        let current_version = self
            .stores
            .policy
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);

        if state.version > current_version {
            self.stores.policy.insert(key, state.clone(), actor.clone());
            debug!(
                "Applied remote policy state update: {} (version: {} -> {})",
                state.model_id, current_version, state.version
            );
        } else {
            debug!(
                "Skipped remote policy state update: {} (version {} <= current {})",
                state.model_id, state.version, current_version
            );
        }
    }

    /// Update rate-limit hash ring with current membership
    pub fn update_rate_limit_membership(&self) {
        // Get all alive nodes from membership store
        let all_members = self.stores.membership.all();
        let alive_nodes: Vec<String> = all_members
            .values()
            .filter(|m| m.status == NodeStatus::Alive as i32)
            .map(|m| m.name.clone())
            .collect();

        self.stores.rate_limit.update_membership(&alive_nodes);
        debug!(
            "Updated rate-limit hash ring with {} alive nodes",
            alive_nodes.len()
        );
    }

    /// Handle node failure and transfer rate-limit ownership
    pub fn handle_node_failure(&self, failed_nodes: &[String]) {
        if failed_nodes.is_empty() {
            return;
        }

        debug!("Handling node failure for rate-limit: {:?}", failed_nodes);

        // Check which keys need ownership transfer
        let affected_keys = self
            .stores
            .rate_limit
            .check_ownership_transfer(failed_nodes);

        if !affected_keys.is_empty() {
            debug!(
                "Ownership transfer needed for {} rate-limit keys",
                affected_keys.len()
            );

            // Update membership to reflect node failures
            self.update_rate_limit_membership();

            // For each affected key, we may need to initialize counters if we're now an owner
            for key in &affected_keys {
                if self.stores.rate_limit.is_owner(key) {
                    debug!("This node is now owner of rate-limit key: {}", key);
                    // Counter will be created on first inc() call
                }
            }
        }
    }

    /// Sync rate-limit counter increment (only if this node is an owner)
    pub fn sync_rate_limit_inc(&self, key: String, delta: i64) {
        if !self.stores.rate_limit.is_owner(&key) {
            // Not an owner, skip
            return;
        }

        self.stores
            .rate_limit
            .inc(key.clone(), self.self_name.clone(), delta);
        debug!("Synced rate-limit increment: key={}, delta={}", key, delta);
    }

    /// Apply remote rate-limit counter update (merge CRDT)
    pub fn apply_remote_rate_limit_counter(
        &self,
        key: String,
        counter: &super::crdt::SyncPNCounter,
    ) {
        // Merge counter regardless of ownership (for CRDT consistency)
        self.stores.rate_limit.merge_counter(key.clone(), counter);
        debug!("Applied remote rate-limit counter update: key={}", key);
    }

    /// Get rate-limit value (aggregate from all owners)
    pub fn get_rate_limit_value(&self, key: &str) -> Option<i64> {
        self.stores.rate_limit.value(key)
    }

    /// Get global rate limit configuration from AppStore
    pub fn get_global_rate_limit_config(&self) -> Option<RateLimitConfig> {
        let key = SKey::new(GLOBAL_RATE_LIMIT_KEY.to_string());
        self.stores
            .app
            .get(&key)
            .and_then(|app_state| serde_json::from_slice::<RateLimitConfig>(&app_state.value).ok())
    }

    /// Check if global rate limit is exceeded
    /// Returns (is_exceeded, current_count, limit)
    pub fn check_global_rate_limit(&self) -> (bool, i64, u64) {
        let config = self.get_global_rate_limit_config().unwrap_or_default();

        if config.limit_per_second == 0 {
            // Rate limit disabled
            return (false, 0, 0);
        }

        // Increment counter if this node is an owner
        self.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), 1);

        // Get aggregated counter value from all owners
        let current_count = self
            .get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY)
            .unwrap_or(0);

        let is_exceeded = current_count > config.limit_per_second as i64;
        (is_exceeded, current_count, config.limit_per_second)
    }

    /// Reset global rate limit counter (called periodically for time window reset)
    pub fn reset_global_rate_limit_counter(&self) {
        // Reset by decrementing the current value
        // Since we use PNCounter, we can't directly reset, but we can track the window
        // For simplicity, we'll use a time-based approach where counters are reset periodically
        // The actual reset logic will be handled by the window manager
        let current_count = self
            .get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY)
            .unwrap_or(0);

        if current_count > 0 {
            // Decrement by current count to effectively reset
            // Note: This is a workaround since PNCounter doesn't support direct reset
            // In production, you might want to use a different approach like timestamped counters
            self.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), -current_count);
        }
    }

    /// Sync tree operation to HA stores
    /// This adds a tree operation (insert or remove) to the tree state for a specific model
    pub fn sync_tree_operation(
        &self,
        model_id: String,
        operation: TreeOperation,
    ) -> Result<(), String> {
        let key = SKey::new(tree_state_key(&model_id));

        // Get current tree state or create new one
        let mut tree_state = if let Some(policy_state) = self.stores.policy.get(&key) {
            // Deserialize existing tree state
            serde_json::from_slice::<TreeState>(&policy_state.config)
                .unwrap_or_else(|_| TreeState::new(model_id.clone()))
        } else {
            TreeState::new(model_id.clone())
        };

        // Add the new operation
        tree_state.add_operation(operation);

        // Serialize and store back
        let serialized = serde_json::to_vec(&tree_state)
            .map_err(|e| format!("Failed to serialize tree state: {}", e))?;

        // Get current version if exists
        let current_version = self
            .stores
            .policy
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);
        let new_version = current_version + 1;

        let state = PolicyState {
            model_id: model_id.clone(),
            policy_type: "tree_state".to_string(),
            config: serialized,
            version: new_version,
        };

        let actor = self.self_name.clone();
        self.stores.policy.insert(key, state, actor);
        debug!(
            "Synced tree operation to HA: model={} (version: {})",
            model_id, new_version
        );

        Ok(())
    }

    /// Get tree state for a model from HA stores
    pub fn get_tree_state(&self, model_id: &str) -> Option<TreeState> {
        let key = SKey::new(tree_state_key(model_id));
        self.stores
            .policy
            .get(&key)
            .and_then(|policy_state| serde_json::from_slice::<TreeState>(&policy_state.config).ok())
    }

    /// Apply remote tree operation to local policy
    /// This is called when receiving tree state updates from other nodes
    pub fn apply_remote_tree_operation(
        &self,
        model_id: String,
        tree_state: TreeState,
        actor: Option<String>,
    ) {
        let key = SKey::new(tree_state_key(&model_id));
        let actor = actor.unwrap_or_else(|| "remote".to_string());

        // Check if we should update based on version
        let current_version = self
            .stores
            .policy
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);

        if tree_state.version > current_version {
            // Serialize tree state
            if let Ok(serialized) = serde_json::to_vec(&tree_state) {
                let state = PolicyState {
                    model_id: model_id.clone(),
                    policy_type: "tree_state".to_string(),
                    config: serialized,
                    version: tree_state.version,
                };

                self.stores.policy.insert(key, state, actor.clone());
                debug!(
                    "Applied remote tree state update: model={} (version: {} -> {})",
                    model_id, current_version, tree_state.version
                );
            } else {
                debug!(
                    "Failed to serialize remote tree state for model={}",
                    model_id
                );
            }
        } else {
            debug!(
                "Skipped remote tree state update: model={} (version {} <= current {})",
                model_id, tree_state.version, current_version
            );
        }
    }
}

/// Optional HA sync manager (can be None if HA is not enabled)
pub type OptionalHASyncManager = Option<Arc<HASyncManager>>;
