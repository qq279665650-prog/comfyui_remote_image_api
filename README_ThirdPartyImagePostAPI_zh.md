# Third Party Image POST API 节点说明

这个插件是在 `wanjin123111/ComfyUI_My_Combined_Nodes` 基础上增加的通用第三方图生图 API 节点。把整个文件夹放到 ComfyUI 的 `custom_nodes` 目录后，重启 ComfyUI，即可在 `Nkxx/API` 分类里找到 `Third Party Image POST API`。

## 节点用途

节点会接收 ComfyUI 里的 `IMAGE` 输入，把图片 POST 到第三方 API 平台，然后把第三方模型返回的图片 URL 或 base64 图片解析成 ComfyUI 的 `IMAGE` 输出。

## 主要输入

- `api_url`: 第三方 API 的 POST 地址。
- `api_key`: API-KEY。
- `model`: 模型名称。
- `prompt`: 提示词。
- `aspect_ratio`: 比例，支持 `auto`、`1:1`、`16:9`、`9:16`、`4:3`、`3:4`、`3:2`、`2:3`、`5:4`、`4:5`、`21:9`。
- `resolution`: 图幅，支持 `1K`、`2K`、`4K`。
- `request_mode`: 请求方式。
  - `json_base64`: 把输入图转为 PNG base64，随 JSON 一起发送。
  - `multipart`: 把输入图作为 multipart/form-data 文件上传。
- `auth_mode`: 鉴权方式。
  - `bearer`: Header 使用 `Authorization: Bearer <api_key>`。
  - `x-api-key`: Header 使用 `x-api-key: <api_key>`。
  - `none`: 不自动添加鉴权 Header。
- `image_1` 到 `image_5`: 参考图片输入。

## 高级参数

- `extra_payload_json`: 追加平台需要的 JSON 字段，例如：

```json
{"num_images": 1, "safety_check": false}
```

- `extra_headers_json`: 追加平台需要的 Header，例如：

```json
{"X-Custom-Header": "value"}
```

- `response_image_path`: 如果平台返回结构很特殊，可以指定图片字段路径，例如：

```text
data.0.url
output.images.0
result.image_base64
```

- `poll_url`: 如果平台先返回任务 ID，再异步生成图片，填轮询地址，例如：

```text
https://api.example.com/v1/tasks/{id}
```

- `poll_id_path`: 任务 ID 字段路径，例如 `id`、`task_id`、`data.id`。
- `result_status_path`: 轮询返回里的状态字段路径，例如 `status`、`data.status`。
- `proxy_url`: 代理地址，可留空。
- `fallback_image`: 调用失败时透传的备用图片。

## 输出

- `image`: 第三方模型返回的图片。
- `status`: 调用状态或错误信息。
- `failed`: `0` 表示成功，`1` 表示失败。
- `video_conn`: 保留原插件透传输出。
- `character_id`: 保留原插件透传输出。

## 默认 JSON 请求体

`json_base64` 模式会发送类似下面的 JSON。部分字段可以通过 `extra_payload_json` 覆盖或追加。

```json
{
  "model": "your-model",
  "prompt": "your prompt",
  "aspect_ratio": "16:9",
  "resolution": "1K",
  "image_size": "1K",
  "size": "1365x768",
  "width": 1344,
  "height": 768,
  "images": ["data:image/png;base64,..."],
  "image": "data:image/png;base64,...",
  "image_base64": "data:image/png;base64,..."
}
```

不同第三方平台字段名可能不同。如果平台要求固定字段名，用 `extra_payload_json` 补充字段；如果平台要求完全不同的上传方式，请优先尝试 `multipart`。
