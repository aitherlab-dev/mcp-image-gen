use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize)]
pub struct JsonRpcRequest {
    #[serde(rename = "jsonrpc")]
    pub _jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Value, code: i64, message: String) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(serde_json::json!({
                "code": code,
                "message": message,
            })),
        }
    }

    pub fn tool_result(id: Value, text: String) -> Self {
        Self::success(
            id,
            serde_json::json!({
                "content": [{"type": "text", "text": text}]
            }),
        )
    }

    pub fn tool_error(id: Value, text: String) -> Self {
        Self::success(
            id,
            serde_json::json!({
                "content": [{"type": "text", "text": text}],
                "isError": true
            }),
        )
    }
}
