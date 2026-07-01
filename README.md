# comfyui_remote_image_api

ComfyUI custom node for sending images to a remote image-generation API.

The main node is:

- Category: `Nkxx/API`
- Node: `Third Party Image POST API`

## Default Grsai Flow

The node now defaults to the same flow as the verified `grsai_nano_banana.py` script:

1. Upload input images to Grsai and collect image URLs.
2. POST a task to `https://grsai.dakka.com.cn/v1/draw/nano-banana`.
3. Poll `https://grsai.dakka.com.cn/v1/draw/result` with the returned task id.
4. Parse returned image URLs or base64 image data into ComfyUI `IMAGE`.

Recommended node values:

- `api_url`: `https://grsai.dakka.com.cn`
- `request_mode`: `grsai_nano_banana`
- `auth_mode`: `bearer`
- `model`: `nano-banana-2`
- `aspect_ratio`: `auto`, `16:9`, `9:16`, `4:3`, etc.
- `resolution`: `1K`, `2K`, or `4K`

The model alias `gemini-3.1-flash-image` is automatically mapped to Grsai model `nano-banana-2`.

## GPT Image 2

The same `grsai_nano_banana` request mode also auto-routes GPT Image models:

- `gpt-image-2`
- `gpt-image-2-vip`

When either model is used, the node posts to `https://grsai.dakka.com.cn/v1/draw/completions` and polls `https://grsai.dakka.com.cn/v1/draw/result`.

For GPT Image, the node maps `aspect_ratio` plus `resolution` into Grsai's pixel `aspectRatio` value. For example, `16:9` + `1K` becomes `1672x941` for `gpt-image-2`, and `1280x720` for `gpt-image-2-vip`.

## Wuyin / 速创 NanoBanana

This plugin also supports Wuyin's NanoBanana APIs:

1. Submit a generation task to `https://api.wuyinkeji.com/api/async/image_nanoBanana2` or `https://api.wuyinkeji.com/api/async/image_nanoBanana_pro`.
2. Poll `https://api.wuyinkeji.com/api/async/detail` with the returned task id.
3. Parse returned image URLs or base64 data into ComfyUI `IMAGE`.

Recommended node values:

- `api_url`: `https://api.wuyinkeji.com`
- `request_mode`: `wuyin_nano_banana2` for NanoBanana2, or `wuyin_nano_banana_pro` for NanoBanana Pro
- `model`: `nano-banana-pro` also auto-routes to `image_nanoBanana_pro`
- `auth_mode`: any value is acceptable; the node sends the API key in `Authorization`
- `resolution`: `1K`, `2K`, or `4K`
- `aspect_ratio`: `auto`, `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3`, `5:4`, `4:5`, `21:9`

Wuyin reference images must be public URLs. Put them in `extra_payload_json`, for example:

```json
{"urls":["https://example.com/ref1.jpg","https://example.com/ref2.jpg"]}
```

To query an existing Wuyin task, switch `request_mode` to `wuyin_query_result` and fill `task_id`.

## Inputs

- `api_url`: API base URL. For Grsai, use `https://grsai.dakka.com.cn`.
- `api_key`: Your API key. The key is only read from the node input and is not stored in the plugin.
- `model`: Model name. Default is `nano-banana-2`.
- `prompt`: Text prompt.
- `aspect_ratio`: Output ratio.
- `resolution`: Image size preset.
- `request_mode`: Use `grsai_nano_banana` for the verified Grsai flow.
- `request_mode`: `grsai_nano_banana`, `wuyin_nano_banana2`, `wuyin_nano_banana_pro`, `wuyin_query_result`, `json_base64`, or `multipart`.
- `auth_mode`: Use `bearer` for Grsai.
- `image_1` to `image_5`: Optional reference images.
- `extra_payload_json`: Optional JSON fields to merge into the Grsai task payload.
- `extra_headers_json`: Optional extra HTTP headers.
- `task_id`: Used by `wuyin_query_result` to fetch an existing job.
- `proxy_url`: Optional proxy URL. Leave empty for direct connection.
- `fallback_image`: Optional image returned when the request fails.

## Outputs

- `image`: Generated image.
- `status`: Success or error text.
- `failed`: `0` on success, `1` on failure.
- `video_conn`: Pass-through output kept for compatibility with existing workflows.
- `character_id`: Pass-through output kept for compatibility with existing workflows.

## Fallback Generic Modes

The older generic modes are still available:

- `json_base64`: Sends input images as base64 JSON.
- `multipart`: Sends input images as multipart form files.

For Grsai Nano Banana generation, use `grsai_nano_banana`.
For Wuyin NanoBanana2, use `wuyin_nano_banana2`; for Wuyin NanoBanana Pro, use `wuyin_nano_banana_pro` or set `model` to `nano-banana-pro`. Use `wuyin_query_result` for result lookup only.

## Install

Copy the whole `comfyui_remote_image_api` folder into ComfyUI's `custom_nodes` directory, then restart ComfyUI.

If dependencies are missing, run this in ComfyUI's Python environment:

```bash
pip install -r requirements.txt
```
