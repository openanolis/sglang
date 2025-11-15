//! HA state synchronization module
//!
//! Handles synchronization of worker and policy states across HA cluster nodes

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::{
    crdt::SKey,
    stores::{PolicyState, StateStores, StoreType, WorkerState},
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
}

/// Optional HA sync manager (can be None if HA is not enabled)
pub type OptionalHASyncManager = Option<Arc<HASyncManager>>;
