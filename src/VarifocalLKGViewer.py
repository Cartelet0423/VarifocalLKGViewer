import sys
import os
import math
import json
import pyglet
from pyglet.gl import *
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import pynng
import cbor2 as cbor
import ctypes
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LG_Calibration')

def get_calibration_from_file() -> dict:
    """
    実行ファイルと同じディレクトリにある 'visual.json' からキャリブレーションデータを読み込みます。
    成功した場合、Bridgeから取得したデータと互換性のある形式で辞書を返します。
    """
    try:
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
        
        json_path = os.path.join(application_path, 'visual.json')

        if not os.path.exists(json_path):
            logger.info("'visual.json' が見つかりません。Looking Glass Bridgeを試します。")
            return {}

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        required_keys = ["pitch", "slope", "center", "DPI", "screenW", "screenH"]
        if not all(key in data for key in required_keys):
            logger.warning("'visual.json' に必要なキーが不足しています。Bridgeにフォールバックします。")
            return {}

        calibration_data = {}
        for key in required_keys:
            value = data[key]
            if isinstance(value, dict) and 'value' in value:
                calibration_data[key] = value['value']
            else:
                calibration_data[key] = value
        
        logger.info("'visual.json' からキャリブレーションを正常に読み込みました。")
        serial = data.get("serial", "local_file_device")
        return {
            serial: {
                "calibration": calibration_data
            }
        }

    except json.JSONDecodeError:
        logger.error("'visual.json' の解析に失敗しました。Bridgeにフォールバックします。")
        return {}
    except Exception as e:
        logger.error(f"'visual.json' 読み込み中に予期せぬエラーが発生しました: {e}")
        return {}

def get_calibration_data_as_dict() -> dict:
    """Looking Glass Bridgeからキャリブレーションデータを取得し、ディクショナリで返します。"""
    address = 'ipc:///tmp/holoplay-driver.ipc'
    timeout = 5000
    client_name = "CalibrationDataExtractor"
    
    try:
        with pynng.Req0(recv_timeout=timeout, send_timeout=timeout) as socket:
            socket.dial(address, block=True)
            
            init_command = {'cmd': {'init': {'appid': client_name}}, 'bin': ''}
            socket.send(cbor.dumps(init_command))
            init_data = cbor.loads(socket.recv())
            if init_data['error'] != 0:
                logger.error(f"Bridgeの初期化に失敗しました: {init_data}")
                return {}
            
            info_command = {'cmd': {'info': {}}, 'bin': ''}
            socket.send(cbor.dumps(info_command))
            device_data = cbor.loads(socket.recv())
            if device_data['error'] != 0:
                logger.error(f"デバイス情報の取得に失敗しました: {device_data}")
                return {}
                
            devices = [d for d in device_data['devices'] if d['state'] == "ok"]
            result = {}
            for device in devices:
                device.pop('state', None)
                calibration_data = device.get('calibration', {})
                device['calibration'] = {
                    key: value['value'] if isinstance(value, dict) else value 
                    for key, value in calibration_data.items()
                }
                result[device.get('hwid', f'unknown_{len(result)}')] = device
            logger.info("Looking Glass Bridgeからキャリブレーションを正常に取得しました。")
            return result
            
    except pynng.exceptions.ConnectionRefused:
        logger.error("Looking Glass Bridgeが実行されていません。")
    except Exception as e:
        logger.error(f"Bridgeとの通信中にエラーが発生しました: {e}")
        
    return {}

VIEWER_VERTEX_SHADER_CODE = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 ScreenCoord;
out vec2 QuiltUvTransformed;
uniform float u_zoom;
uniform vec2 u_pan;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    ScreenCoord = aTexCoord;
    QuiltUvTransformed = (aTexCoord - vec2(0.5)) / u_zoom + vec2(0.5) + u_pan;
}
"""
VIEWER_FRAGMENT_SHADER_CODE = FRAGMENT_SHADER_CODE = """
#version 330 core
in vec2 ScreenCoord;
in vec2 QuiltUvTransformed;
out vec4 FragColor;
uniform sampler2D _MainTex;
uniform float _Pitch;
uniform float _DPI;
uniform float _Slope;
uniform float _Center;
uniform int _DisplayWidth;
uniform int _DisplayHeight;
uniform float _Focus;
uniform vec2 u_windowResolution;
uniform bool u_useOverrideSettings;
uniform int u_quiltWidth;
uniform int u_quiltHeight;
uniform float u_aspect;
uniform float u_maxAngleScale;

float fmod(float x, float y) {
    return x - y * trunc(x/y);
}
uint DecodeUInt16FromRegion(sampler2D tex, int columnIndex)
{
    uint decodedValue = 0u;
    for (int rowIndex = 0; rowIndex < 8; ++rowIndex)
    {
        float local_uv_x = (float(columnIndex) + 0.5) / 5.0;
        float local_uv_y = 1.0 - ((float(rowIndex) + 0.5) / 8.0);
        float global_uv_x = 0.975 + (local_uv_x / 40.0);
        float global_uv_y = 0.960 + (local_uv_y / 25.0);
        vec4 encodedColor = texture(tex, vec2(global_uv_x, global_uv_y));
        uint bitG = (encodedColor.g > 0.5) ? 1u : 0u;
        uint bitB = (encodedColor.b > 0.5) ? 1u : 0u;
        int bitIndexB = rowIndex * 2;
        int bitIndexG = bitIndexB + 1;
        decodedValue |= (bitG << uint(bitIndexG));
        decodedValue |= (bitB << uint(bitIndexB));
    }
    return decodedValue;
}
vec3 GenerateCoordinateMap(vec2 screenPos)
{
    float adjustedPitch = (_DPI * 3.0 / _Pitch) / cos(atan(1.0 / _Slope));
    float x = screenPos.x * float(_DisplayWidth-1) * 3.0;
    float y = (1.0 - screenPos.y) * float(_DisplayHeight-1);
    float cr = fmod(x - (y + adjustedPitch * _Center) / (_Slope / 3.0), adjustedPitch) / adjustedPitch;
    float cg = fmod(x+1.0 - (y + adjustedPitch * _Center) / (_Slope / 3.0), adjustedPitch) / adjustedPitch;
    float cb = fmod(x+2.0 - (y + adjustedPitch * _Center) / (_Slope / 3.0), adjustedPitch) / adjustedPitch;
    return vec3(cr, cg, cb);
}
float CalculateAngleScale(float viewIndex, int totalViews, float maxAngleScale)
{
    if (totalViews <= 1) return 1.0;
    float centerViewIndex = float(totalViews - 1) / 2.0;
    if (centerViewIndex == 0.0) return 1.0;
    float normalizedIndex = viewIndex / centerViewIndex - 1.0;
    return 1.0 + (abs(normalizedIndex) * (maxAngleScale - 1.0));
}
vec2 AdjustUVWithFocus(vec2 uv, float angleScale, float viewIndex, int totalViews)
{
    if (totalViews <= 1) return uv;
    float centerViewIndex = float(totalViews - 1) / 2.0;
    if (centerViewIndex == 0.0) return uv;
    float normalizedIndex = (viewIndex - centerViewIndex) / centerViewIndex;
    vec2 centeredUV = uv - vec2(0.5, 0.5);
    centeredUV.x = centeredUV.x / angleScale;
    float shift = (_Focus - 0.5) * (1.0 - 1.0/angleScale) * sign(normalizedIndex);
    centeredUV.x += shift;
    return centeredUV + vec2(0.5, 0.5);
}
void main()
{
    int QuiltWidth;
    int QuiltHeight;
    float Aspect;
    float MaxAngleScale;
    if (u_useOverrideSettings) {
        QuiltWidth = u_quiltWidth;
        QuiltHeight = u_quiltHeight;
        Aspect = u_aspect;
        MaxAngleScale = u_maxAngleScale;
    } else {
        uint val0 = DecodeUInt16FromRegion(_MainTex, 0);
        uint val1 = DecodeUInt16FromRegion(_MainTex, 1);
        uint val2 = DecodeUInt16FromRegion(_MainTex, 2);
        uint val3 = DecodeUInt16FromRegion(_MainTex, 3);
        QuiltWidth = int(val0);
        QuiltHeight = int(val1);
        Aspect = float(val2) / 16383.0;
        MaxAngleScale = float(val3) / 16383.0;
    }
    float imageAspectHW = Aspect;
    if (imageAspectHW <= 0.0) { imageAspectHW = 9.0/16.0; }
    float windowAspectWH = u_windowResolution.x / u_windowResolution.y;
    float imageAspectWH = 1.0 / imageAspectHW;
    vec2 scale = vec2(1.0, 1.0);
    if (windowAspectWH > imageAspectWH) {
        scale.x = imageAspectWH / windowAspectWH;
    } else {
        scale.y = windowAspectWH / imageAspectWH;
    }
    vec2 centered_quilt = QuiltUvTransformed - 0.5;
    vec2 scaled_quilt = centered_quilt / scale;
    vec2 uv_quilt = scaled_quilt + 0.5;
    if (uv_quilt.x < 0.0 || uv_quilt.x > 1.0 || uv_quilt.y < 0.0 || uv_quilt.y > 1.0) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec3 coordMap = GenerateCoordinateMap(ScreenCoord);
    vec3 viewIndices = vec3(float(QuiltWidth * QuiltHeight - 1) - round(coordMap * float(QuiltWidth * QuiltHeight - 1)));
    int internal_ViewCount = QuiltWidth * QuiltHeight - 1;
    if (QuiltWidth <= 0 || QuiltHeight <= 0 || internal_ViewCount <= 0) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float Focus = _Focus;
    float angleScaleR = CalculateAngleScale(viewIndices.r, internal_ViewCount, MaxAngleScale);
    float angleScaleG = CalculateAngleScale(viewIndices.g, internal_ViewCount, MaxAngleScale);
    float angleScaleB = CalculateAngleScale(viewIndices.b, internal_ViewCount, MaxAngleScale);
    float xr = fmod(viewIndices.r, float(QuiltWidth)) / float(QuiltWidth);
    float yr = floor(viewIndices.r / float(QuiltWidth)) / float(QuiltHeight);
    float xg = fmod(viewIndices.g, float(QuiltWidth)) / float(QuiltWidth);
    float yg = floor(viewIndices.g / float(QuiltWidth)) / float(QuiltHeight);
    float xb = fmod(viewIndices.b, float(QuiltWidth)) / float(QuiltWidth);
    float yb = floor(viewIndices.b / float(QuiltWidth)) / float(QuiltHeight);
    vec2 quiltUvR = AdjustUVWithFocus(uv_quilt, angleScaleR, viewIndices.r, internal_ViewCount);
    vec2 quiltUvG = AdjustUVWithFocus(uv_quilt, angleScaleG, viewIndices.g, internal_ViewCount);
    vec2 quiltUvB = AdjustUVWithFocus(uv_quilt, angleScaleB, viewIndices.b, internal_ViewCount);
    quiltUvR = vec2(quiltUvR.x/float(QuiltWidth), quiltUvR.y/float(QuiltHeight));
    quiltUvG = vec2(quiltUvG.x/float(QuiltWidth), quiltUvG.y/float(QuiltHeight));
    quiltUvB = vec2(quiltUvB.x/float(QuiltWidth), quiltUvB.y/float(QuiltHeight));
    vec2 sampleUvR = vec2(xr + quiltUvR.x, yr + quiltUvR.y);
    vec2 sampleUvG = vec2(xg + quiltUvG.x, yg + quiltUvG.y);
    vec2 sampleUvB = vec2(xb + quiltUvB.x, yb + quiltUvB.y);
    sampleUvR = clamp(sampleUvR, vec2(xr, yr), vec2(xr + 1.0/float(QuiltWidth), yr + 1.0/float(QuiltHeight)));
    sampleUvG = clamp(sampleUvG, vec2(xg, yg), vec2(xg + 1.0/float(QuiltWidth), yg + 1.0/float(QuiltHeight)));
    sampleUvB = clamp(sampleUvB, vec2(xb, yb), vec2(xb + 1.0/float(QuiltWidth), yb + 1.0/float(QuiltHeight)));
    vec4 color_linear;
    color_linear.r = texture(_MainTex, sampleUvR).r;
    color_linear.g = texture(_MainTex, sampleUvG).g;
    color_linear.b = texture(_MainTex, sampleUvB).b;
    color_linear.a = 1.0;
    FragColor = color_linear;
}
"""

# ### NEW: Quilt画像リマップ用シェーダー ###
REMAP_VERTEX_SHADER_CODE = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""
REMAP_FRAGMENT_SHADER_CODE = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D u_inputQuilt;
uniform float u_focus;         
uniform int u_quiltWidth;      
uniform int u_quiltHeight;     
uniform float u_maxAngleScale; 

float calculateAngleScale(float viewIndex, int totalViews, float maxAngleScale) {
    if (totalViews <= 1) return 1.0;
    float centerView = float(totalViews - 1) / 2.0;
    if (centerView == 0.0) return 1.0;
    float normalizedIndex = (viewIndex / centerView) - 1.0;
    return 1.0 + (abs(normalizedIndex) * (maxAngleScale - 1.0));
}

vec2 getRefocusedSubUV(vec2 subUV, float angleScale, float viewIndex, int totalViews) {
    if (totalViews <= 1) return subUV;
    float centerView = float(totalViews - 1) / 2.0;
    if (centerView == 0.0) return subUV;
    vec2 centeredUV = subUV - 0.5;
    centeredUV.x /= angleScale;
    float normalizedIndex = (viewIndex - centerView) / centerView;
    float shift = (u_focus - 0.5) * (1.0 - 1.0 / angleScale) * sign(normalizedIndex);
    centeredUV.x += shift;
    return centeredUV + 0.5;
}

void main() {
    int totalViews = u_quiltWidth * u_quiltHeight - 1;
    if (totalViews < 2) {
        FragColor = texture(u_inputQuilt, TexCoord);
        return;
    }

    vec2 quiltDim = vec2(u_quiltWidth, u_quiltHeight);

    vec2 quiltCoords = TexCoord * quiltDim;
    vec2 viewRowCol = floor(quiltCoords);
    vec2 subUV = fract(quiltCoords);      

    float viewIndex = viewRowCol.y * quiltDim.x + viewRowCol.x;

    float samplingViewIndex = viewIndex;
    if (int(viewIndex) == totalViews - 1) {
        samplingViewIndex = float(totalViews - 2);
    }

    float angleScale = calculateAngleScale(samplingViewIndex, totalViews, u_maxAngleScale);

    vec2 refocusedSubUV = getRefocusedSubUV(subUV, angleScale, samplingViewIndex, totalViews);

    float samplingViewCol = mod(samplingViewIndex, quiltDim.x);
    float samplingViewRow = floor(samplingViewIndex / quiltDim.x);
    vec2 samplingViewRowCol = vec2(samplingViewCol, samplingViewRow);
    vec2 sourceUV = (samplingViewRowCol + refocusedSubUV) / quiltDim;

    FragColor = texture(u_inputQuilt, sourceUV);
}
"""

def create_shader_program(vertex_source, fragment_source):
    vertex_shader = pyglet.graphics.shader.Shader(vertex_source, 'vertex')
    fragment_shader = pyglet.graphics.shader.Shader(fragment_source, 'fragment')
    return pyglet.graphics.shader.ShaderProgram(vertex_shader, fragment_shader)

class LKGViewerWindow(pyglet.window.Window):
    def __init__(self, calibration_params, tk_root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shader_program = create_shader_program(VIEWER_VERTEX_SHADER_CODE, VIEWER_FRAGMENT_SHADER_CODE)
        self.remap_shader_program = create_shader_program(REMAP_VERTEX_SHADER_CODE, REMAP_FRAGMENT_SHADER_CODE)
        
        self.calibration_params = calibration_params

        self.tk_root = tk_root
        pyglet.clock.schedule_interval(self.update_tk, 1/120.0)

        positions = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
        tex_coords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        indices = [0, 1, 2, 0, 2, 3]

        self.vertex_list = self.shader_program.vertex_list_indexed(
            4, GL_TRIANGLES, indices, aPos=('f', positions), aTexCoord=('f', tex_coords)
        )
        
        self.texture = None
        self.current_image_path = None
        self.image_data = None
        self.focus_cache = self.load_cache('focus_cache.json')
        self.quilt_settings_cache = self.load_cache('quilt_settings_cache.json')
        self.view_cache = self.load_cache('view_cache.json')
        self.override_settings = {}
        self.quilt_params = {}
        self.focus_value = 0.5
        self.zoom = 1.0
        self.pan_x, self.pan_y = 0.0, 0.0
        self.is_dragging = False
        self.key_states = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.key_states)
        self.focus_change_speed = 0.01
        self.settings_window_instance = None
        self.last_non_varifocal_settings = {}
        self.DEFAULT_VARIFOCAL_MAS = 1.1
        pyglet.clock.schedule_interval(self.update_focus, 1/60.0)
        self.setup_shader_uniforms()

    def update_tk(self, dt):
        if self.tk_root:
            self.tk_root.update()

    def setup_shader_uniforms(self):
        params = self.calibration_params
        self.shader_program['_Pitch'] = float(params.get('pitch', -234))
        self.shader_program['_DPI'] = float(params.get('DPI', 491))
        self.shader_program['_Slope'] = float(params.get('slope', -6.6))
        self.shader_program['_Center'] = float(params.get('center', 0.0))
        self.shader_program['_DisplayWidth'] = int(params.get('screenW', 1440))
        self.shader_program['_DisplayHeight'] = int(params.get('screenH', 2560))

    def update_focus(self, dt):
        focus_updated = False
        if self.key_states[pyglet.window.key.UP]:
            self.focus_value = min(1.0, self.focus_value + self.focus_change_speed)
            focus_updated = True
        elif self.key_states[pyglet.window.key.DOWN]:
            self.focus_value = max(0.0, self.focus_value - self.focus_change_speed)
            focus_updated = True
        if focus_updated and self.current_image_path:
            self.focus_cache[self.current_image_path] = self.focus_value
            self.set_caption(f"VarifocalLKGViewer - Focus: {self.focus_value:.2f} (Manual)")

    def decode_quilt_params(self):
        if not self.image_data:
            self.quilt_params = {}
            return
        data = self.image_data.get_data()
        width, height = self.image_data.width, self.image_data.height
        fmt, pitch = self.image_data.format, self.image_data.pitch
        format_size = len(fmt)
        try:
            g_index, b_index = fmt.index('G'), fmt.index('B')
        except ValueError:
            logger.error(f"Image format '{fmt}' not supported for quilt decoding.")
            self.quilt_params = {}
            return
        decoded_values = []
        for col_idx in range(5):
            val = 0
            for row_idx in range(8):
                local_uv_x, local_uv_y = ((col_idx + 0.5) / 5.0, 1.0 - ((row_idx + 0.5) / 8.0))
                global_uv_x, global_uv_y = (0.975 + (local_uv_x / 40.0), 0.960 + (local_uv_y / 25.0))
                px, py = int(global_uv_x * width), int(global_uv_y * height)
                if not (0 <= px < width and 0 <= py < height): continue
                try:
                    idx = py * pitch + px * format_size
                    bit_g = 1 if data[idx + g_index] > 127 else 0
                    bit_b = 1 if data[idx + b_index] > 127 else 0
                    bit_idx_b, bit_idx_g = row_idx * 2, row_idx * 2 + 1
                    val |= (bit_g << bit_idx_g) | (bit_b << bit_idx_b)
                except IndexError:
                    continue
            decoded_values.append(val)
        if len(decoded_values) >= 4:
            self.quilt_params = {
                'QuiltWidth': decoded_values[0],
                'QuiltHeight': decoded_values[1],
                'Aspect': decoded_values[2] / 16383.0,
                'MaxAngleScale': decoded_values[3] / 16383.0,
            }
        else:
            self.quilt_params = {}

    def load_texture(self, image_path):
        if not os.path.exists(image_path):
            print(f"画像ファイルが見つかりません: {image_path}")
            return
        try:
            image = pyglet.image.load(image_path)
            self.texture = image.get_texture()
            self.image_data = image.get_image_data()
            for param in [GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER]:
                glTexParameteri(self.texture.target, param, GL_NEAREST)
            for param in [GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T]:
                glTexParameteri(self.texture.target, param, GL_CLAMP_TO_EDGE)
            self.current_image_path = image_path
            self.focus_value = self.focus_cache.get(self.current_image_path, 0.5)
            self._load_quilt_settings()
            self._load_view_settings()
            current_params = self.override_settings or self.quilt_params
            if current_params and current_params.get('MaxAngleScale', 0.0) <= 1.00001:
                self.last_non_varifocal_settings = current_params.copy()
            if self.settings_window_instance and self.settings_window_instance.winfo_exists():
                self.settings_window_instance.update_fields(current_params)
            self.set_caption(f"VarifocalLKGViewer - Focus: {self.focus_value:.2f}")
        except Exception as e:
            print(f"テクスチャ読み込み中にエラーが発生しました: {e}")
            self.image_data, self.quilt_params = None, {}

    def _load_quilt_settings(self):
        self.decode_quilt_params()
        cached_settings = self.quilt_settings_cache.get(self.current_image_path)
        if cached_settings:
            self.override_settings = cached_settings
        else:
            self.override_settings = self.quilt_params.copy()
            if self.override_settings:
                if self.override_settings.get('MaxAngleScale', 1.0) <= 1.00001:
                    self.override_settings['MaxAngleScale'] = self.DEFAULT_VARIFOCAL_MAS
                self.quilt_settings_cache[self.current_image_path] = self.override_settings

    def _load_view_settings(self):
        view_settings = self.view_cache.get(self.current_image_path)
        if view_settings:
            self.zoom = view_settings.get("zoom", 1.0)
            self.pan_x = view_settings.get("pan_x", 0.0)
            self.pan_y = view_settings.get("pan_y", 0.0)
        else:
            self.zoom = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0

    def _update_view_cache(self):
        if not self.current_image_path:
            return
        view_settings = {
            "zoom": self.zoom,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y
        }
        self.view_cache[self.current_image_path] = view_settings

    def load_cache(self, filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self, data, filename):
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"キャッシュ '{filename}' の保存に失敗しました: {e}")

    def on_draw(self):
        self.clear()
        if not self.texture: return
        self.shader_program.use()
        glBindTexture(self.texture.target, self.texture.id)
        self.shader_program['_MainTex'] = 0
        self.shader_program['_Focus'] = self.focus_value
        self.shader_program['u_zoom'] = self.zoom
        self.shader_program['u_pan'] = (self.pan_x, self.pan_y)
        self.shader_program['u_windowResolution'] = (self.width, self.height)
        if self.override_settings:
            self.shader_program['u_useOverrideSettings'] = True
            self.shader_program['u_quiltWidth'] = int(self.override_settings.get('QuiltWidth', 0))
            self.shader_program['u_quiltHeight'] = int(self.override_settings.get('QuiltHeight', 0))
            self.shader_program['u_aspect'] = float(self.override_settings.get('Aspect', 1.0))
            self.shader_program['u_maxAngleScale'] = float(self.override_settings.get('MaxAngleScale', 1.0))
        else:
            self.shader_program['u_useOverrideSettings'] = False
        self.vertex_list.draw(GL_TRIANGLES)
        self.shader_program.stop()

    def change_image(self, direction):
        if not self.current_image_path: return
        try:
            directory = os.path.dirname(self.current_image_path)
            current_filename = os.path.basename(self.current_image_path)
            valid_ext = ['.png', '.jpg', '.jpeg', '.bmp']
            files = sorted([f for f in os.listdir(directory) if any(f.lower().endswith(e) for e in valid_ext)])
            if not files: return
            idx = (files.index(current_filename) + direction + len(files)) % len(files)
            self.load_texture(os.path.join(directory, files[idx]))
        except (FileNotFoundError, ValueError, Exception) as e:
            print(f"画像切り替え中にエラーが発生しました: {e}")

    def set_fullscreen(self, fullscreen):
        if fullscreen:
            display = pyglet.display.get_display()
            screens = display.get_screens()
            lkg_screen = screens[0]
            w, h = self.calibration_params.get('screenW', 1440), self.calibration_params.get('screenH', 2560)
            for s in screens:
                if s.width == w and s.height == h:
                    lkg_screen = s; break
            super().set_fullscreen(True, screen=lkg_screen)
        else:
            super().set_fullscreen(False)

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Open Image File", filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*"))
        )
        if file_path: self.load_texture(file_path)

    def clamp_pan(self):
        max_pan_val = 0.5 - (0.5 / self.zoom)
        if self.zoom <= 1.0: self.pan_x, self.pan_y = 0.0, 0.0; return
        self.pan_x = max(min(self.pan_x, max_pan_val), -max_pan_val)
        self.pan_y = max(min(self.pan_y, max_pan_val), -max_pan_val)

    def on_key_press(self, symbol, modifiers):
        if symbol in (pyglet.window.key.F11, pyglet.window.key.F): self.set_fullscreen(not self.fullscreen)
        elif symbol == pyglet.window.key.RIGHT: self.change_image(1)
        elif symbol == pyglet.window.key.LEFT: self.change_image(-1)
        elif symbol == pyglet.window.key.ESCAPE: self.close()
        elif symbol == pyglet.window.key.O and (modifiers & pyglet.window.key.MOD_CTRL):
            self.open_file_dialog()
        elif symbol == pyglet.window.key.P and (modifiers & pyglet.window.key.MOD_CTRL):
            self.open_settings_window()
        elif symbol == pyglet.window.key.S and (modifiers & pyglet.window.key.MOD_CTRL):
            self.save_refocused_quilt()
        elif symbol == pyglet.window.key.R: 
            self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
            self._update_view_cache()


    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.is_dragging = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            self.is_dragging = True
            self.pan_x -= dx / (self.width * self.zoom)
            self.pan_y -= dy / (self.height * self.zoom)
            self.clamp_pan()
            self._update_view_cache()

    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT and not self.is_dragging:
            if not self.image_data: return
            q_params = self.override_settings or self.quilt_params
            quilt_width = q_params.get('QuiltWidth')
            quilt_height = q_params.get('QuiltHeight')
            if not quilt_width or not quilt_height:
                logger.warning("Quilt dimensions not found. Cannot set focus from depth.")
                return
            u, v = x / self.width, y / self.height
            depth_map_u = (u + float(quilt_width - 1)) / float(quilt_width)
            depth_map_v = (v + float(quilt_height - 1)) / float(quilt_height)
            tex_width, tex_height = self.image_data.width, self.image_data.height
            px = max(0, min(tex_width - 1, int(depth_map_u * tex_width)))
            py = max(0, min(tex_height - 1, int(depth_map_v * tex_height)))
            try:
                data, pitch, fmt = self.image_data.get_data(), self.image_data.pitch, self.image_data.format
                r_index = fmt.index('R')
                idx = py * pitch + len(fmt) * px
                r_byte = data[idx + r_index]
                depth = r_byte / 255.0
                new_focus = 1.0 - pow(depth, 2.2)
                self.focus_value = max(0.0, min(1.0, new_focus))
                if self.current_image_path:
                    self.focus_cache[self.current_image_path] = self.focus_value
                self.set_caption(f"VarifocalLKGViewer - Focus: {self.focus_value:.2f} (Clicked)")
            except (IndexError, ValueError) as e:
                logger.error(f"Error getting depth at ({px}, {py}): {e}")

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if not self.texture: return
        zoom_factor = 1.1 if scroll_y > 0 else 1 / 1.1
        old_zoom = self.zoom
        self.zoom *= zoom_factor
        if self.zoom < 1.0: self.zoom = 1.0
        elif self.zoom > 30.0: self.zoom = 30.0
        if self.zoom == old_zoom: return
        mx, my = x / self.width, y / self.height
        pan_dx = (mx - 0.5) * (1.0 / old_zoom - 1.0 / self.zoom)
        pan_dy = (my - 0.5) * (1.0 / old_zoom - 1.0 / self.zoom)
        self.pan_x += pan_dx
        self.pan_y += pan_dy
        self.clamp_pan()
        self._update_view_cache()

    def save_refocused_quilt(self):
        """現在のフォーカス値を適用してQuilt画像を再生成し、ファイルに保存します。"""
        if not self.texture:
            messagebox.showwarning("警告", "保存する画像が読み込まれていません。")
            return

        current_params = self.override_settings or self.quilt_params
        q_width = current_params.get('QuiltWidth')
        q_height = current_params.get('QuiltHeight')
        max_angle_scale = current_params.get('MaxAngleScale', 1.0)
        
        if not all([q_width, q_height, max_angle_scale]):
            messagebox.showerror("エラー", "Quiltパラメータが不完全なため、画像を保存できません。")
            return

        initial_dir = os.path.dirname(self.current_image_path)
        base, ext = os.path.splitext(os.path.basename(self.current_image_path))
        initial_file = f"{base}_refocused_{self.focus_value:.3f}.png"
        output_path = filedialog.asksaveasfilename(
            initialdir=initial_dir, initialfile=initial_file, defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if not output_path:
            return

        logger.info(f"Refocused quiltを {output_path} に保存しています...")

        W, H = self.texture.width, self.texture.height
        fbo = GLuint()
        glGenFramebuffers(1, ctypes.byref(fbo))
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        color_tex = GLuint()
        glGenTextures(1, ctypes.byref(color_tex))
        glBindTexture(GL_TEXTURE_2D, color_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            logger.error("FBOの作成に失敗しました。")
            return

        prev_viewport = (GLint * 4)()
        glGetIntegerv(GL_VIEWPORT, prev_viewport)
        glViewport(0, 0, W, H)

        self.remap_shader_program.use()
        glBindTexture(self.texture.target, self.texture.id)
        
        self.remap_shader_program['u_inputQuilt'] = 0
        self.remap_shader_program['u_focus'] = self.focus_value
        self.remap_shader_program['u_quiltWidth'] = int(q_width)
        self.remap_shader_program['u_quiltHeight'] = int(q_height)
        self.remap_shader_program['u_maxAngleScale'] = float(max_angle_scale)
        
        self.vertex_list.draw(GL_TRIANGLES)
        self.remap_shader_program.stop()

        buffer = (GLubyte * (W * H * 4))()
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, buffer)
        image = Image.frombytes('RGBA', (W, H), buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(output_path)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteTextures(1, ctypes.byref(color_tex))
        glDeleteFramebuffers(1, ctypes.byref(fbo))
        glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3])

        logger.info(f"画像を正常に保存しました: {output_path}")
        messagebox.showinfo("成功", f"画像を下記に保存しました:\n{output_path}")

    def on_close(self):
        self.save_cache(self.focus_cache, 'focus_cache.json')
        self.save_cache(self.quilt_settings_cache, 'quilt_settings_cache.json')
        self.save_cache(self.view_cache, 'view_cache.json')
        pyglet.clock.unschedule(self.update_focus)
        pyglet.clock.unschedule(self.update_tk)
        super().on_close()

    def open_settings_window(self):
        if self.settings_window_instance and self.settings_window_instance.winfo_exists():
            self.settings_window_instance.lift()
            return
        if not self.current_image_path:
            messagebox.showwarning("警告", "画像が読み込まれていません。")
            return
        current_settings = self.override_settings or self.quilt_params
        if not current_settings:
            messagebox.showerror("エラー", "Quiltパラメータを特定できません。")
            return
        root = tk.Toplevel(self.tk_root)
        self.settings_window_instance = root
        root.title("Quilt 設定 (Live)")
        root.attributes("-topmost", True)
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vars = {
            'QuiltWidth': tk.StringVar(), 'QuiltHeight': tk.StringVar(),
            'Aspect': tk.StringVar(), 'MaxAngleScale': tk.StringVar()
        }
        varifocal_var = tk.BooleanVar()
        trace_ids = {}

        def _apply_live_settings(*args):
            try:
                new_settings = {key: float(var.get()) for key, var in vars.items()}
                new_settings['QuiltWidth'] = int(new_settings['QuiltWidth'])
                new_settings['QuiltHeight'] = int(new_settings['QuiltHeight'])
                self.override_settings = new_settings
                self.quilt_settings_cache[self.current_image_path] = new_settings
            except (ValueError, TypeError):
                pass

        def _remove_traces():
            for key, var in vars.items():
                if key in trace_ids and trace_ids[key]: var.trace_remove("write", trace_ids[key])
            if 'varifocal' in trace_ids and trace_ids['varifocal']: varifocal_var.trace_remove("write", trace_ids['varifocal'])
            
        def _add_traces():
            for key, var in vars.items():
                trace_ids[key] = var.trace_add("write", _apply_live_settings)
            trace_ids['varifocal'] = varifocal_var.trace_add("write", toggle_varifocal)

        def populate_fields(source_data):
            _remove_traces()
            vars['QuiltWidth'].set(str(source_data.get('QuiltWidth', '')))
            vars['QuiltHeight'].set(str(source_data.get('QuiltHeight', '')))
            vars['Aspect'].set(f"{source_data.get('Aspect', 0.0):.5f}")
            mas = source_data.get('MaxAngleScale', 1.0)
            vars['MaxAngleScale'].set(f"{mas:.5f}")
            varifocal_var.set(mas > 1.00001)
            _add_traces()
        root.update_fields = populate_fields
        
        def toggle_varifocal(*args):
            if not varifocal_var.get():
                if self.last_non_varifocal_settings:
                    populate_fields(self.last_non_varifocal_settings)
                else:
                    vars['MaxAngleScale'].set("1.00000")
            _apply_live_settings()

        def reset_to_image_defaults():
            self.decode_quilt_params()
            if self.current_image_path in self.quilt_settings_cache:
                del self.quilt_settings_cache[self.current_image_path]
            self.override_settings = self.quilt_params.copy()
            populate_fields(self.quilt_params)
            _apply_live_settings()
            
        def on_settings_close():
            self.settings_window_instance = None
            root.destroy()
        root.protocol("WM_DELETE_WINDOW", on_settings_close)
        _add_traces()
        fields = [("Quilt Width:", 'QuiltWidth'), ("Quilt Height:", 'QuiltHeight'),
                  ("Aspect (H/W):", 'Aspect'), ("Max Angle Scale:", 'MaxAngleScale')]
        for i, (label, key) in enumerate(fields):
            ttk.Label(main_frame, text=label).grid(column=0, row=i, sticky=tk.W, pady=2)
            ttk.Entry(main_frame, textvariable=vars[key]).grid(column=1, row=i, sticky=(tk.W, tk.E))
        ttk.Checkbutton(main_frame, text="可変焦点 (Varifocal)", variable=varifocal_var).grid(
                        column=0, row=len(fields), columnspan=2, sticky=tk.W, pady=5)
        ttk.Button(main_frame, text="画像の値にリセット", command=reset_to_image_defaults).grid(
                        column=0, row=len(fields) + 1, columnspan=2, sticky="ew", pady=10)
        populate_fields(current_settings)

def create_cursor(width: int, height: int):
    image_data = bytearray(width * height * 4)
    center_x = width / 2.0
    center_y = height / 2.0
    max_distance = width / 2.0

    for y in range(height):
        for x in range(width):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            normalized_distance = min(distance / max_distance, 1.0)
            alpha_float = math.cos(normalized_distance * (math.pi/2))**2
            alpha_byte = int(alpha_float * 255)
            index = (y * width + x) * 4
            
            image_data[index] = 255
            image_data[index + 1] = 255
            image_data[index + 2] = 255
            image_data[index + 3] = alpha_byte
    image = pyglet.image.ImageData(width, height, 'RGBA', bytes(image_data))
    return pyglet.window.ImageMouseCursor(image, width // 2, height // 2)

if __name__ == '__main__':
    tk_root = tk.Tk()
    tk_root.withdraw()
    calibration_data = get_calibration_from_file()
    if not calibration_data:
        calibration_data = get_calibration_data_as_dict()
    if not calibration_data:
        messagebox.showerror(
            "Initialization Error",
            "キャリブレーションデータを 'visual.json' または Looking Glass Bridge から取得できませんでした。\n"
            "'visual.json' を確認するか、Bridgeを起動してください。"
        )
        sys.exit()
    first_device_calibration = list(calibration_data.values())[0].get('calibration', {})
    config = pyglet.gl.Config(major_version=3, minor_version=3, double_buffer=True)
    window = LKGViewerWindow(
        calibration_params=first_device_calibration,
        tk_root=tk_root,
        caption="VarifocalLKGViewer", resizable=True, config=config, width=225, height=400
    )
    window.set_mouse_cursor(create_cursor(64, 64))
    pyglet.app.run()