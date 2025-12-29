//! gRPC client for SGLang encoder workers in EPD (Encode-Prefill-Decode) mode
//!
//! This client communicates with encode workers that process multimodal inputs
//! (images, videos, audio) and send embeddings to prefill workers.

use std::time::Duration;

use tonic::{transport::Channel, Request};
use tracing::debug;

// Include the generated protobuf code
#[allow(clippy::all)]
pub mod proto {
    #![allow(clippy::all, unused_qualifications)]
    tonic::include_proto!("sglang.grpc.encoder");
}

/// gRPC client for SGLang encoder workers
#[derive(Clone)]
pub struct SglangEncoderClient {
    client: proto::sglang_encoder_client::SglangEncoderClient<Channel>,
}

impl SglangEncoderClient {
    /// Create a new client and connect to the encoder worker
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Connecting to SGLang encoder at {}", endpoint);

        // Convert grpc:// to http:// for tonic
        let http_endpoint = if let Some(addr) = endpoint.strip_prefix("grpc://") {
            format!("http://{}", addr)
        } else {
            endpoint.to_string()
        };

        let channel = Channel::from_shared(http_endpoint)?
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_timeout(Duration::from_secs(10))
            .keep_alive_while_idle(true)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .tcp_nodelay(true)
            .connect()
            .await?;

        let client = proto::sglang_encoder_client::SglangEncoderClient::new(channel);

        Ok(Self { client })
    }

    /// Send an encode request to the encoder worker
    ///
    /// The encoder processes multimodal items and sends embeddings directly
    /// to the prefill worker via ZMQ.
    pub async fn encode(
        &self,
        req: proto::EncodeRequest,
    ) -> Result<proto::EncodeResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!(
            req_id = %req.req_id,
            mm_items_count = req.mm_items.len(),
            num_parts = req.num_parts,
            part_idx = req.part_idx,
            prefill_host = %req.prefill_host,
            "Sending gRPC encode request"
        );

        let mut client = self.client.clone();
        let request = Request::new(req);

        let response = client.encode(request).await?;

        debug!("Encode request completed successfully");
        Ok(response.into_inner())
    }

    /// Send embeddings to prefill server (mooncake backend)
    pub async fn send(
        &self,
        req: proto::SendRequest,
    ) -> Result<proto::SendResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!(
            req_id = %req.req_id,
            prefill_host = %req.prefill_host,
            embedding_port = req.embedding_port,
            "Sending gRPC send request"
        );

        let mut client = self.client.clone();
        let request = Request::new(req);

        let response = client.send(request).await?;
        Ok(response.into_inner())
    }

    /// Register scheduler receive URL (zmq_to_scheduler backend)
    pub async fn scheduler_receive_url(
        &self,
        req: proto::SchedulerReceiveUrlRequest,
    ) -> Result<proto::SchedulerReceiveUrlResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!(
            req_id = %req.req_id,
            receive_url = %req.receive_url,
            receive_count = req.receive_count,
            "Sending gRPC scheduler_receive_url request"
        );

        let mut client = self.client.clone();
        let request = Request::new(req);

        let response = client.scheduler_receive_url(request).await?;
        Ok(response.into_inner())
    }

    /// Build an encode request
    pub fn build_encode_request(
        mm_items: Vec<String>,
        req_id: String,
        num_parts: i32,
        part_idx: i32,
        prefill_host: String,
        embedding_port: Vec<i32>,
    ) -> proto::EncodeRequest {
        proto::EncodeRequest {
            mm_items,
            req_id,
            num_parts,
            part_idx,
            prefill_host,
            embedding_port,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_request_construction() {
        let req = SglangEncoderClient::build_encode_request(
            vec!["https://example.com/image.jpg".to_string()],
            "test-req-123".to_string(),
            1,
            0,
            "localhost".to_string(),
            vec![8998],
        );

        assert_eq!(req.req_id, "test-req-123");
        assert_eq!(req.mm_items.len(), 1);
        assert_eq!(req.num_parts, 1);
        assert_eq!(req.part_idx, 0);
        assert_eq!(req.prefill_host, "localhost");
        assert_eq!(req.embedding_port, vec![8998]);
    }

    #[tokio::test]
    async fn test_client_connect_invalid_endpoint() {
        let result = SglangEncoderClient::connect("invalid://endpoint").await;
        assert!(result.is_err());
    }
}
