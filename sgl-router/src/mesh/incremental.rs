//! Incremental update collection and batching
//!
//! Collects local state changes and batches them for efficient transmission

use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use parking_lot::RwLock;
use tracing::{debug, trace};

use super::{
    crdt::SKey,
    gossip::StateUpdate,
    stores::{StateStores, StoreType},
};

/// Tracks the last sent version for each key in each store
#[derive(Debug, Clone, Default)]
struct LastSentVersions {
    worker: HashMap<String, u64>,
    policy: HashMap<String, u64>,
    app: HashMap<String, u64>,
    membership: HashMap<String, u64>,
    rate_limit: HashMap<String, u64>, // Track last sent timestamp for rate limit counters
}

/// Incremental update collector
pub struct IncrementalUpdateCollector {
    stores: Arc<StateStores>,
    self_name: String,
    last_sent: Arc<RwLock<LastSentVersions>>,
}

impl IncrementalUpdateCollector {
    pub fn new(stores: Arc<StateStores>, self_name: String) -> Self {
        Self {
            stores,
            self_name,
            last_sent: Arc::new(RwLock::new(LastSentVersions::default())),
        }
    }

    /// Generic helper function to collect updates for stores that use serialization
    fn collect_serialized_updates<T: serde::Serialize>(
        all_items: std::collections::BTreeMap<SKey, T>,
        versions: HashMap<String, u64>,
        self_name: &str,
        last_sent_map: &mut HashMap<String, u64>,
        log_message: &str,
        log_field: impl Fn(&T) -> String,
    ) -> Vec<StateUpdate> {
        let mut updates = Vec::new();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        for (key, state) in all_items {
            let key_str = key.as_str().to_string();
            let current_version = versions.get(&key_str).copied().unwrap_or(0);
            let last_sent_version = last_sent_map.get(&key_str).copied().unwrap_or(0);

            if current_version > last_sent_version {
                if let Ok(serialized) = serde_json::to_vec(&state) {
                    updates.push(StateUpdate {
                        key: key_str.clone(),
                        value: serialized,
                        version: current_version,
                        actor: self_name.to_string(),
                        timestamp,
                    });

                    last_sent_map.insert(key_str.clone(), current_version);
                    trace!(
                        "{}: {} (version: {})",
                        log_message,
                        log_field(&state),
                        current_version
                    );
                }
            }
        }

        updates
    }

    /// Generic helper function to collect updates for stores that don't need serialization
    fn collect_direct_updates<T>(
        all_items: std::collections::BTreeMap<SKey, T>,
        versions: HashMap<String, u64>,
        self_name: &str,
        last_sent_map: &mut HashMap<String, u64>,
        get_value: impl Fn(&T) -> Vec<u8>,
        log_message: &str,
        log_field: impl Fn(&T) -> String,
    ) -> Vec<StateUpdate> {
        let mut updates = Vec::new();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        for (key, state) in all_items {
            let key_str = key.as_str().to_string();
            let current_version = versions.get(&key_str).copied().unwrap_or(0);
            let last_sent_version = last_sent_map.get(&key_str).copied().unwrap_or(0);

            if current_version > last_sent_version {
                updates.push(StateUpdate {
                    key: key_str.clone(),
                    value: get_value(&state),
                    version: current_version,
                    actor: self_name.to_string(),
                    timestamp,
                });

                last_sent_map.insert(key_str.clone(), current_version);
                trace!(
                    "{}: {} (version: {})",
                    log_message,
                    log_field(&state),
                    current_version
                );
            }
        }

        updates
    }

    /// Collect incremental updates for a specific store type
    pub fn collect_updates_for_store(&self, store_type: StoreType) -> Vec<StateUpdate> {
        let mut updates = Vec::new();
        let mut last_sent = self.last_sent.write();

        match store_type {
            StoreType::Worker => {
                use super::stores::WorkerState;
                let all_workers = self.stores.worker.all();
                // Collect versions first to avoid borrowing issues
                let versions: HashMap<String, u64> = all_workers
                    .keys()
                    .map(|key| {
                        let key_str = key.as_str().to_string();
                        let version = self
                            .stores
                            .worker
                            .get_metadata(key)
                            .map(|(v, _)| v)
                            .unwrap_or_default();
                        (key_str, version)
                    })
                    .collect();
                updates.extend(Self::collect_serialized_updates(
                    all_workers,
                    versions,
                    &self.self_name,
                    &mut last_sent.worker,
                    "Collected worker update",
                    |state: &WorkerState| state.worker_id.clone(),
                ));
            }
            StoreType::Policy => {
                use super::stores::PolicyState;
                let all_policies = self.stores.policy.all();
                // Collect versions first to avoid borrowing issues
                let versions: HashMap<String, u64> = all_policies
                    .keys()
                    .map(|key| {
                        let key_str = key.as_str().to_string();
                        let version = self
                            .stores
                            .policy
                            .get_metadata(key)
                            .map(|(v, _)| v)
                            .unwrap_or(0);
                        (key_str, version)
                    })
                    .collect();
                updates.extend(Self::collect_serialized_updates(
                    all_policies,
                    versions,
                    &self.self_name,
                    &mut last_sent.policy,
                    "Collected policy update",
                    |state: &PolicyState| state.model_id.clone(),
                ));
            }
            StoreType::App => {
                use super::stores::AppState;
                let all_apps = self.stores.app.all();
                // Collect versions first to avoid borrowing issues
                let versions: HashMap<String, u64> = all_apps
                    .keys()
                    .map(|key| {
                        let key_str = key.as_str().to_string();
                        let version = self
                            .stores
                            .app
                            .get_metadata(key)
                            .map(|(v, _)| v)
                            .unwrap_or(0);
                        (key_str, version)
                    })
                    .collect();
                updates.extend(Self::collect_direct_updates(
                    all_apps,
                    versions,
                    &self.self_name,
                    &mut last_sent.app,
                    |state: &AppState| state.value.clone(),
                    "Collected app update",
                    |state: &AppState| state.key.clone(),
                ));
            }
            StoreType::Membership => {
                use super::stores::MembershipState;
                let all_members = self.stores.membership.all();
                // Collect versions first to avoid borrowing issues
                let versions: HashMap<String, u64> = all_members
                    .keys()
                    .map(|key| {
                        let key_str = key.as_str().to_string();
                        let version = self
                            .stores
                            .membership
                            .get_metadata(key)
                            .map(|(v, _)| v)
                            .unwrap_or(0);
                        (key_str, version)
                    })
                    .collect();
                updates.extend(Self::collect_serialized_updates(
                    all_members,
                    versions,
                    &self.self_name,
                    &mut last_sent.membership,
                    "Collected membership update",
                    |state: &MembershipState| state.name.clone(),
                ));
            }
            StoreType::RateLimit => {
                // Collect rate limit counters from owners
                let rate_limit_keys = self.stores.rate_limit.keys();
                let mut last_sent = self.last_sent.write();

                for key in rate_limit_keys {
                    // Only collect if this node is an owner
                    if self.stores.rate_limit.is_owner(&key) {
                        if let Some(counter) = self.stores.rate_limit.get_counter(&key) {
                            // Use timestamp as version for rate limit counters
                            let current_timestamp = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_nanos()
                                as u64;

                            let last_sent_timestamp =
                                last_sent.rate_limit.get(&key).copied().unwrap_or(0);

                            // Only send if enough time has passed (to avoid too frequent updates)
                            // Or if this is the first time
                            // Check if at least 1 second has passed since last send
                            if current_timestamp > last_sent_timestamp + 1_000_000_000 {
                                // Serialize the counter snapshot
                                if let Ok(serialized) = serde_json::to_vec(&counter.snapshot()) {
                                    let key_str = key.clone();
                                    updates.push(StateUpdate {
                                        key: key_str.clone(),
                                        value: serialized,
                                        version: current_timestamp,
                                        actor: self.self_name.clone(),
                                        timestamp: current_timestamp,
                                    });

                                    // Update last sent timestamp
                                    last_sent.rate_limit.insert(key_str, current_timestamp);
                                    trace!("Collected rate limit counter update: {}", key);
                                }
                            }
                        }
                    }
                }
            }
        }

        debug!(
            "Collected {} incremental updates for store {:?}",
            updates.len(),
            store_type
        );
        updates
    }

    /// Collect all incremental updates across all stores
    pub fn collect_all_updates(&self) -> Vec<(StoreType, Vec<StateUpdate>)> {
        let mut all_updates = Vec::new();

        for store_type in [
            StoreType::Worker,
            StoreType::Policy,
            StoreType::App,
            StoreType::Membership,
            StoreType::RateLimit,
        ] {
            let updates = self.collect_updates_for_store(store_type);
            if !updates.is_empty() {
                all_updates.push((store_type, updates));
            }
        }

        all_updates
    }

    /// Mark updates as sent (called after successful transmission)
    pub fn mark_sent(&self, store_type: StoreType, updates: &[StateUpdate]) {
        let mut last_sent = self.last_sent.write();
        match store_type {
            StoreType::Worker => {
                for update in updates {
                    last_sent.worker.insert(update.key.clone(), update.version);
                }
            }
            StoreType::Policy => {
                for update in updates {
                    last_sent.policy.insert(update.key.clone(), update.version);
                }
            }
            StoreType::App => {
                for update in updates {
                    last_sent.app.insert(update.key.clone(), update.version);
                }
            }
            StoreType::Membership => {
                for update in updates {
                    last_sent
                        .membership
                        .insert(update.key.clone(), update.version);
                }
            }
            StoreType::RateLimit => {
                for update in updates {
                    last_sent
                        .rate_limit
                        .insert(update.key.clone(), update.version);
                }
            }
        }
    }
}
