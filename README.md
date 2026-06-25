# comfyui_remote_image_api

ComfyUI custom nodes based on `wanjin123111/ComfyUI_My_Combined_Nodes`, with an added generic remote image-generation API node.

新增节点：`Third Party Image POST API`

分类：`Nkxx/API`

## 功能

这个节点可以把 ComfyUI 中传入的图片 POST 到第三方 API 平台，调用远程图生图大模型，再把返回的图片 URL 或 base64 图片解析为 ComfyUI `IMAGE` 输出。

适合需要在节点内填写这些参数的工作流：

- API 平台访问地址
- API-KEY
- 模型名称
- 比例：`16:9`、`4:3`、`9:16`、`1:1` 等
- 图幅：`1K`、`2K`、`4K`
- 输入图片：`image_1` 到 `image_5`

## 安装

把整个 `comfyui_remote_image_api` 文件夹放到 ComfyUI 的 `custom_nodes` 目录，然后重启 ComfyUI。

如果缺少依赖，在 ComfyUI 的 Python 环境中执行：

```bash
pip install -r requirements.txt
```

## 节点输入

- `api_url`: 第三方 API 的 POST 地址。
- `api_key`: API-KEY。
- `model`: 模型名称。
- `prompt`: 提示词。
- `aspect_ratio`: 比例，支持 `auto`、`1:1`、`16:9`、`9:16`、`4:3`、`3:4`、`3:2`、`2:3`、`5:4`、`4:5`、`21:9`。
- `resolution`: 图幅，支持 `1K`、`2K`、`4K`。
- `request_mode`: 请求方式。
  - `json_base64`: 将图片以 PNG base64 放入 JSON 请求体。
  - `multipart`: 将图片作为 `multipart/form-data` 文件上传。
- `auth_mode`: 鉴权方式。
  - `bearer`: `Authorization: Bearer <api_key>`
  - `x-api-key`: `x-api-key: <api_key>`
  - `none`: 不自动添加鉴权 Header。
- `image_1` 到 `image_5`: 参考图片输入。

## 高级参数

- `extra_payload_json`: 追加第三方平台需要的 JSON 字段。
- `extra_headers_json`: 追加第三方平台需要的 Header。
- `response_image_path`: 指定返回图片字段路径，例如 `data.0.url`、`output.images.0`、`result.image_base64`。
- `poll_url`: 异步任务轮询地址，例如 `https://api.example.com/v1/tasks/{id}`。
- `poll_id_path`: 异步任务 ID 字段路径，例如 `id`、`task_id`、`data.id`。
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

`json_base64` 模式会发送类似下面的 JSON。字段可用 `extra_payload_json` 覆盖或追加。

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

不同第三方平台字段名可能不同。如果平台要求固定字段名，可以用 `extra_payload_json` 补充或覆盖字段；如果平台要求表单文件上传，可以切换为 `multipart`。

## 说明

本插件仅作为 ComfyUI 扩展工具。调用第三方 API 产生的费用、内容和合规责任由使用者自行承担。
