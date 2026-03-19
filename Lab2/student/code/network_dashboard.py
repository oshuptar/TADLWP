import os
import shutil
import base64

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import plotly.express as px
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from PIL import Image
import sys

# Add Lab1 code directory to path to import experiment_logger
lab1_code_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Lab1', 'teacher', 'code')
if lab1_code_path not in sys.path:
    sys.path.insert(0, lab1_code_path)

from experiment_logger import ExperimentRegistry
from experiment_logger_dense import ExperimentWithDense


COLORMAP_NAME = 'managua_r'
NUM_NEURONS_THAT_TRIGGERS_VISUALIZATION_SWITCH = 10


class ActivationVisualizer:
    def __init__(self, temp_dir='streamlit_temp'):
        self.temp_dir = temp_dir
        self.special_filename_to_signal_done = 'render_2d_activation_visualization_done'
        self.colormap = mpl.colormaps[COLORMAP_NAME]
        self.colormap_norm = plt.Normalize(vmin=-1.0, vmax=1.0)

    def render_2d_activation_visualization(self, dirname, model, central_x, X, y, dim_1, dim_2, dim_1_values, dim_2_values):
        dir = os.path.join(self.temp_dir, dirname)
        layer_image_paths = self._load_rendered(dir)
        if layer_image_paths and len(layer_image_paths) > 0:
            for image_paths in layer_image_paths:
                yield image_paths
            return

        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir, exist_ok=True)
        activations = self._calculate_values_of_all_activations(model, central_x, dim_1, dim_2, dim_1_values, dim_2_values)
        num_layers = len(activations[0][0])

        for layer_i in range(num_layers-1):
            layer_image_paths = []
            for neuron_i in range(len(activations[0][0][layer_i])):
                Z = [[
                    activations[i1][i2][layer_i][neuron_i].item() for i2 in list(range(len(dim_2_values)))
                    ] for i1 in list(range(len(dim_1_values)))
                ]
                filename = os.path.join(dir, f'layer_{layer_i}_neuron_{neuron_i}')
                filename_with_extension = self._render_contour_plot(Z, filename)
                layer_image_paths.append(filename_with_extension)
            yield layer_image_paths

        last_layer_image_paths = []
        for neuron_i in range(len(activations[0][0][-1])):
            Z = [[
                activations[i1][i2][-1][neuron_i].item() for i2 in range(len(dim_2_values))
                ] for i1 in range(len(dim_1_values))
            ]
            filename = os.path.join(dir, f'layer_{num_layers-1}_neuron_{neuron_i}')
            self._render_contour_plot_with_scatter(
                dim_1_values, dim_2_values, Z,
                X[:, dim_1], X[:, dim_2], y[:, neuron_i],
                zmin=-1.0, zmax=1.0, out_filename=filename
            )
            last_layer_image_paths.append(filename + '.png')
        yield last_layer_image_paths
        with open(os.path.join(dir, self.special_filename_to_signal_done), 'w') as f:
            f.write('Done!')

    def _calculate_values_of_all_activations(self, model, central_x, dim_1, dim_2, dim_1_values, dim_2_values):
        activations = [
                [[] for i2 in range(len(dim_1_values))]
                for i1 in range(len(dim_2_values))
        ]
        for i1, val_1 in enumerate(dim_1_values):
            for i2, val_2 in enumerate(dim_2_values):
                x = central_x.clone()
                x[:, dim_1] = val_1
                x[:, dim_2] = val_2
                output = x
                activations[i1][i2].append(output[0])
                for m, module in enumerate(model):
                    output = module(output)
                    if m+1<len(model) and model[m+1].__module__ == 'torch.nn.modules.activation':
                        # next module is an activation, let's skip visualizing the current one
                        continue
                    activations[i1][i2].append(output[0])
        return activations

    def _load_rendered(self, dir):
        try:
            filenames = os.listdir(dir)
        except FileNotFoundError:
            return None
        if len(filenames) == 0:
            return None
        if not os.path.isfile(
            os.path.join(dir, self.special_filename_to_signal_done)
        ):
            return None
        layer_image_paths = []
        for filename in filenames:
            if 'layer' not in filename or '_neuron_' not in filename:
                continue
            try:
                layer_idx = int(filename.split('_')[1])
            except IndexError:
                continue
            except ValueError:
                continue
            while len(layer_image_paths) < layer_idx + 1:
                layer_image_paths.append([])
            layer_image_paths[layer_idx].append(os.path.join(dir, filename))
        return layer_image_paths

    def _render_contour_plot(self, Z, out_filename):
        im = Image.fromarray((self.colormap(self.colormap_norm(np.rot90(Z)))*255).astype(np.uint8))
        out_filename_with_extension = out_filename + '.bmp'
        im.save(out_filename_with_extension)
        return out_filename_with_extension

    def _render_contour_plot_with_scatter(self, X, Y, Z, x, y, z, zmin, zmax, out_filename):
        xmin, xmax, ymin, ymax = X[0], X[-1], Y[0], Y[-1]
        fig, ax = plt.subplots(figsize=(3,3))
        im = ax.imshow(
            np.rot90(Z),
            cmap=COLORMAP_NAME,
            vmin=zmin,
            vmax=zmax,
            extent=[xmin, xmax, ymin, ymax],
            alpha = 0.9,
            interpolation='bicubic'
        )
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        if x.shape[0] > 100:
            indices = list(range(x.shape[0]))
            np.random.shuffle(indices)
            indices = indices[:100]
            x = x[indices]
            y = y[indices]
            z = z[indices]
        ax.scatter(x, y, s=10, c=z, cmap='Set1', vmin=zmin, vmax=zmax)

        ax.axis('off')
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        plt.savefig(out_filename, bbox_inches='tight')
        plt.close()


def _image_to_data_url(image_path):
    """Convert local image file to base64 data URL. Returns original path if it's a URL."""
    if image_path.startswith(('http://', 'https://', 'data:')):
        return image_path

    if not os.path.exists(image_path):
        st.warning(f"Image file not found: {image_path}")
        return image_path

    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()

        ext = Path(image_path).suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.svg': 'image/svg+xml'
        }.get(ext, 'image/jpeg')

        return f"data:{mime_type};base64,{data}"
    except Exception as e:
        st.warning(f"Error loading image {image_path}: {e}")
        return image_path


def display_weights_as_paths(epoch, num_layers, height, last_batch_scale, x_positions, layer_to_gap):
    weights = load_weights_state(experiment_name, epoch)
    biases = load_biases_state(experiment_name, epoch)
    colormap = mpl.colormaps[COLORMAP_NAME]
    colormap_norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    svg_content = ''
    for layer_idx in range(num_layers):
        is_next_last = (layer_idx + 1 == num_layers)
        width = height
        next_height = int(height * last_batch_scale) if is_next_last else height

        x1 = x_positions[layer_idx] + width
        x2 = x_positions[layer_idx + 1]

        for input_idx in range(get_num_layer_inputs(experiment_name, epoch, layer_idx)):
            y1 = 10 + layer_to_gap[layer_idx]//2 + input_idx * (height + layer_to_gap[layer_idx]) + 0.35 * height

            for output_idx in range(get_num_layer_outputs(experiment_name, epoch, layer_idx)):
                y2 = 10 + layer_to_gap[layer_idx+1]//2 + output_idx * (next_height + layer_to_gap[layer_idx+1]) + 0.35 * next_height
                weight_value = weights[layer_idx][output_idx][input_idx]

                line_color = mpl.colors.to_hex(
                    colormap(colormap_norm(
                        weight_value
                    ))
                )
                stroke_width = int(min(12, max(3, abs(weight_value) * 6)))
                stroke_opacity = min(1.0, abs(weight_value))

                control_offset = (x2 - x1) * 0.5
                path = f'M {x1} {y1} C {x1 + control_offset} {y1}, {x2 - control_offset} {y2}, {x2} {y2}'

                weight_id = f'layer_{layer_idx}_input_{input_idx}_output_{output_idx}'
                svg_content += f'<path class="connection-line-display" d="{path}" default-stroke="{line_color}" stroke="{line_color}" stroke-width="{stroke_width}" default-stroke-opacity="{stroke_opacity}" stroke-opacity="{stroke_opacity}" fill="none" style="cursor: pointer; pointer-events: none;" visualization-weight-id="{weight_id}"/>\n'
                svg_content += f'<path class="connection-line-hitbox" d="{path}" stroke="{line_color}" stroke-width="{stroke_width+6}" fill="none" style="cursor: pointer;" weight-id="{weight_id}"><title>{weight_value:.2f}</title></path>\n'

        for output_idx in range(get_num_layer_outputs(experiment_name, epoch, layer_idx)):
            yb = 10 + layer_to_gap[layer_idx+1]//2 + output_idx * (next_height + layer_to_gap[layer_idx+1]) + next_height / 2 + height // 3
            xb = x2 - 10
            weight_value = biases[layer_idx][output_idx]
            line_color = mpl.colors.to_hex(
                colormap(colormap_norm(
                    weight_value
                ))
            )
            stroke_width = 9
            stroke_opacity = min(1.0, 0.15 + 0.85 * abs(weight_value))

            path = f'M {xb} {yb} h 10'
            weight_id = f'layer_{layer_idx}_output_{output_idx}'
            svg_content += f'<path class="connection-line-display" d="{path}" default-stroke="{line_color}" stroke="{line_color}" stroke-width="{stroke_width}" default-stroke-opacity="{stroke_opacity}" stroke-opacity="{stroke_opacity}" fill="none" style="cursor: pointer; pointer-events: none;" visualization-weight-id="{weight_id}"/>\n'
            svg_content += f'<path class="connection-line-hitbox" d="{path}" stroke="{line_color}" stroke-width="{stroke_width+6}" fill="none" style="cursor: pointer;" weight-id="{weight_id}"><title>{weight_value:.2f}</title></path>\n'
    return svg_content


def display_weights_as_one_line(epoch, num_layers, height, last_batch_scale, x_positions, layer_to_gap, layer_to_num_columns):
    weights = load_weights_state(experiment_name, epoch)
    biases = load_biases_state(experiment_name, epoch)
    colormap = mpl.colormaps[COLORMAP_NAME]
    colormap_norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    svg_content = ''
    for layer_idx in range(num_layers):
        is_next_last = (layer_idx + 1 == num_layers)
        width = height
        next_height = int(height * last_batch_scale) if is_next_last else height

        x1 = x_positions[layer_idx] + width
        x2 = x_positions[layer_idx + 1]

        y1 = 10 + layer_to_gap[layer_idx]//2 + get_num_layer_inputs(experiment_name, epoch, layer_idx) / layer_to_num_columns[layer_idx] // 2 * height + 0.35 * height
        y2 = 10 + layer_to_gap[layer_idx+1]//2 + get_num_layer_outputs(experiment_name, epoch, layer_idx) / layer_to_num_columns[layer_idx+1] // 2 * next_height + 0.35 * next_height

        weight_strength = weights[layer_idx].flatten().std()
        line_color = mpl.colors.to_hex(
            colormap(colormap_norm(
                weight_strength
            ))
        )
        stroke_width = 20.0
        stroke_opacity = 1.0

        control_offset = (x2 - x1) * 0.5
        path = f'M {x1} {y1} C {x1 + control_offset} {y1}, {x2 - control_offset} {y2}, {x2} {y2}'
        title = f'{len(weights[layer_idx].flatten())} weights. mean: {weights[layer_idx].flatten().mean():.3f}, std: {weights[layer_idx].flatten().std():.3f}'

        weight_id = f'layer_{layer_idx}_weights'
        svg_content += f'<path class="connection-line-display" d="{path}" default-stroke="{line_color}" stroke="{line_color}" stroke-width="{stroke_width}" default-stroke-opacity="{stroke_opacity}" stroke-opacity="{stroke_opacity}" fill="none" style="cursor: pointer; pointer-events: none;" visualization-weight-id="{weight_id}"/>\n'
        svg_content += f'<path class="connection-line-hitbox" d="{path}" stroke="{line_color}" stroke-width="{stroke_width+6}" fill="none" style="cursor: pointer;" weight-id="{weight_id}"><title>{title}</title></path>\n'

        yb = 10 + layer_to_gap[layer_idx+1]//2 + get_num_layer_outputs(experiment_name, epoch, layer_idx) / layer_to_num_columns[layer_idx+1] // 2  * next_height + next_height / 2 + height // 3
        xb = x2 - 10
        line_color = mpl.colors.to_hex(
            colormap(colormap_norm(
                biases[layer_idx].std()
            ))
        )
        stroke_width = 20.0
        stroke_opacity = 1.0

        title = f'{len(biases[layer_idx])} biases. mean: {biases[layer_idx].mean():.3f}, std: {biases[layer_idx].std():.3f}'
        path = f'M {xb} {yb} h 10'
        weight_id = f'layer_{layer_idx}_biases'
        svg_content += f'<path class="connection-line-display" d="{path}" default-stroke="{line_color}" stroke="{line_color}" stroke-width="{stroke_width}" default-stroke-opacity="{stroke_opacity}" stroke-opacity="{stroke_opacity}" fill="none" style="cursor: pointer; pointer-events: none;" visualization-weight-id="{weight_id}"/>\n'
        svg_content += f'<path class="connection-line-hitbox" d="{path}" stroke="{line_color}" stroke-width="{stroke_width+6}" fill="none" style="cursor: pointer;" weight-id="{weight_id}"><title>{title}</title></path>\n'

    return svg_content


def network_visualization_component(experiment_name, epoch, height=200, gap=10, batch_gap=20, last_batch_scale=1.0, scroll_key=None):
    num_layers = get_num_layers(experiment_name, epoch)
    layer_to_num_images = [get_num_layer_inputs(experiment_name, epoch, layer_idx) for layer_idx in range(num_layers)]
    max_images = max(layer_to_num_images)
    if max_images < NUM_NEURONS_THAT_TRIGGERS_VISUALIZATION_SWITCH:
        layer_to_num_columns = [1 for num_images in layer_to_num_images]
    else:
        layer_to_num_columns = [max(1, int(np.cbrt(num_images))) for num_images in layer_to_num_images]
    layer_to_num_columns.append(1)
    img_width = height

    x_positions = []
    current_x = 10
    for layer_idx in range(num_layers + 1):
        x_positions.append(current_x)
        if layer_idx == num_layers:
            current_x += int(img_width * last_batch_scale) + batch_gap
        else:
            num_columns = layer_to_num_columns[layer_idx]
            current_x += img_width * num_columns + num_columns - 1 + batch_gap

    total_width = current_x + 10

    last_layer_height = int(height * last_batch_scale)
    num_outputs_last_layer = get_num_layer_outputs(experiment_name, epoch, num_layers-1)
    last_layer_height_total = num_outputs_last_layer * last_layer_height + (num_outputs_last_layer - 1) * int(gap * last_batch_scale)
    if max_images < NUM_NEURONS_THAT_TRIGGERS_VISUALIZATION_SWITCH:
        other_layers_height = max_images * height + (max_images - 1) * gap
    else:
        other_layers_height = max_images // max(layer_to_num_columns) * height + (max_images - 1)
    total_height_val = max(last_layer_height_total, other_layers_height) + 20
    layer_to_gap = [
        (total_height_val - layer_to_num_images[layer_idx] // layer_to_num_columns[layer_idx] * height) / (layer_to_num_images[layer_idx] + 1)
        for layer_idx in range(num_layers)
    ]
    layer_to_gap.append(gap)

    svg_content = ''
    if max_images < NUM_NEURONS_THAT_TRIGGERS_VISUALIZATION_SWITCH:
        svg_content += display_weights_as_paths(epoch, num_layers, height, last_batch_scale, x_positions, layer_to_gap)
    else:
        svg_content += display_weights_as_one_line(epoch, num_layers, height, last_batch_scale, x_positions, layer_to_gap, layer_to_num_columns)

    layer_image_paths = st.session_state.layer_image_paths[epoch]
    for layer_idx in range(num_layers+1):
        x = x_positions[layer_idx]
        is_last_batch = (layer_idx == num_layers)

        image_height = int(height * last_batch_scale) if is_last_batch else height
        image_width = int(img_width * last_batch_scale) if is_last_batch else img_width

        for image_idx, image_path in enumerate(layer_image_paths[layer_idx]):
            if max_images < NUM_NEURONS_THAT_TRIGGERS_VISUALIZATION_SWITCH:
                y = 10 + layer_to_gap[layer_idx]//2 + image_idx * (image_height + layer_to_gap[layer_idx])
            else:
                y = 10 + layer_to_gap[layer_idx]//2 + (image_idx // layer_to_num_columns[layer_idx]) * (image_height + 1)
            x_plus_column = x + (image_width + 1) * (image_idx % layer_to_num_columns[layer_idx])
            svg_content += f'''
            <image 
                href="{image_path}" 
                x="{x_plus_column}" 
                y="{y}"
                width="{image_width}" 
                height="{image_height}"
                style="border-radius: 8px; image-rendering: pixelated;"/>
            '''

    html_code = f"""
    <style>
        .center-wrapper {{
            display: flex;
            justify-content: center;
            width: 100%;
        }}

        .connection-line-hitbox {{
            opacity: 0;
            transition: opacity 0.1s ease-in-out;
        }}

        .connection-line-hitbox:hover {{
            opacity: 1;
        }}

        .gallery-container {{
            overflow-x: auto;
            overflow-y: hidden;
            padding: 10px 0;
            scroll-behavior: smooth;
        }}
        
        .gallery-container::-webkit-scrollbar {{
            height: 8px;
        }}
        
        .gallery-container::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 10px;
        }}
        
        .gallery-container::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 10px;
        }}
        
        .gallery-container::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}

    </style>

    <div class="center-wrapper"> 
        <div class="gallery-container" id="gallery-{scroll_key}">
            <svg width="{total_width}" height="{total_height_val}" xmlns="http://www.w3.org/2000/svg">
                {svg_content}
            </svg>
        </div>
    </div>

    <script>
        (function() {{
            const container = document.getElementById('gallery-{scroll_key}');
            const scrollKey = 'scroll_pos_{scroll_key}';
            const selectedWeightId = 'selected_weight_id_{scroll_key}';

            const savedScroll = localStorage.getItem(scrollKey);
            if (savedScroll !== null) {{
                container.scrollLeft = parseInt(savedScroll);
            }}

            container.addEventListener('scroll', function() {{
                localStorage.setItem(scrollKey, container.scrollLeft);
            }});

            const connectionLines = document.querySelectorAll('.connection-line-hitbox');
            connectionLines.forEach(line => {{
                line.addEventListener('click', function() {{
                    const weightId = this.getAttribute('weight-id');
                    if (localStorage.getItem(selectedWeightId) !== weightId) {{
                        localStorage.setItem(selectedWeightId, weightId);
                    }}
                    else {{
                        localStorage.setItem(selectedWeightId, null);
                    }}
                    updateColorDisplay(localStorage.getItem(selectedWeightId));
                }});
            }});

            const weightId = localStorage.getItem(selectedWeightId);
            if (weightId) {{
                updateColorDisplay(weightId);
            }}
        }})();

        function updateColorDisplay(weightId) {{
            const allPaths = document.querySelectorAll('.connection-line-display');
            allPaths.forEach(path => {{
                path.style.strokeOpacity = window.getComputedStyle(path).getPropertyValue('default-stroke-opacity');
                path.style.stroke = window.getComputedStyle(path).getPropertyValue('default-stroke');
            }});
            const selectedPath = document.querySelectorAll('path[visualization-weight-id="' + weightId + '"]');
            if (selectedPath) {{
                selectedPath.forEach(path => {{
                    path.style.strokeOpacity = 1.0;
                    path.style.stroke = '#FF4B4B';
                }});
            }}
        }}
    </script>
    """

    total_height = total_height_val + 50
    components.html(html_code, height=total_height, scrolling=False)


@st.cache_data
def get_num_layers(experiment_name, epoch):
    return len(load_weights_state(experiment_name, epoch))


@st.cache_data
def get_num_layer_outputs(experiment_name, epoch, layer_idx):
    weights_state = load_weights_state(experiment_name, epoch)
    return weights_state[layer_idx].shape[0]


@st.cache_data
def get_num_layer_inputs(experiment_name, epoch, layer_idx):
    weights_state = load_weights_state(experiment_name, epoch)
    return weights_state[layer_idx].shape[1]


@st.cache_resource
def load_model(experiment_name, epoch):
    ex = ExperimentWithDense(experiment_name, verbose=False)
    ex.step = epoch
    return ex.load_torch_model_sequential('model')


def load_activation_visualizations(experiment_name, step):
    ex = ExperimentWithDense(experiment_name, verbose=False)
    central_x = torch.zeros(1, 2)
    dim_1_values = np.arange(-1.0, 1.01, 0.05)
    dim_2_values = np.arange(-1.0, 1.01, 0.05)
    ex.step = 0
    train_X = ex.load_npy_array('train_X.npy')
    train_y = ex.load_npy_array('train_y.npy')
    ex.step = step
    model = load_model(experiment_name, step)
    layer_image_paths = []
    activation_visualizer = ActivationVisualizer()
    for image_paths in activation_visualizer.render_2d_activation_visualization(
        os.path.join('activations', experiment_name, str(step)),
        model, central_x, train_X, train_y,
        0, 1, dim_1_values, dim_2_values
    ):
        processed_image_paths = [_image_to_data_url(ip) for ip in image_paths]
        layer_image_paths.append(processed_image_paths)
    st.session_state.layer_image_paths[step] = layer_image_paths


@st.cache_data
def load_weights_state(experiment_name, step):
    model = load_model(experiment_name, step)
    weights_state = []
    for m in range(len(model)):
        # Check for both Linear and Dense layers
        if isinstance(model[m], nn.Linear) or (hasattr(model[m], 'weight') and hasattr(model[m], 'bias') and type(model[m]).__name__ == 'Dense'):
            weights_state.append(model[m].weight.detach().numpy())
    return weights_state


@st.cache_data
def load_biases_state(experiment_name, step):
    model = load_model(experiment_name, step)
    biases_state = []
    for m in range(len(model)):
        # Check for both Linear and Dense layers
        if isinstance(model[m], nn.Linear) or (hasattr(model[m], 'weight') and hasattr(model[m], 'bias') and type(model[m]).__name__ == 'Dense'):
            biases_state.append(model[m].bias.detach().numpy())
    return biases_state


def get_recommended_step_increment(experiment):
    if experiment.last_step < 20:
        return 1
    return experiment.last_step // 10


def load_weight_history(experiment_name, layer_idx, input_idx=None, output_idx=None):
    ex = ExperimentWithDense(experiment_name, verbose=False)
    weight_history = []
    for epoch in range(0, ex.last_step+1, get_recommended_step_increment(ex)):
        weights_state = load_weights_state(experiment_name, epoch)
        if input_idx is None:
            weight_history.append(weights_state[layer_idx])
        else:
            weight_history.append(weights_state[layer_idx][output_idx][input_idx])
    return weight_history


def load_bias_history(experiment_name, layer_idx, output_idx=None):
    ex = ExperimentWithDense(experiment_name, verbose=False)
    bias_history = []
    for epoch in range(0, ex.last_step+1, get_recommended_step_increment(ex)):
        biases_state = load_biases_state(experiment_name, epoch)
        if output_idx is None:
            bias_history.append(biases_state[layer_idx])
        else:
            bias_history.append(biases_state[layer_idx][output_idx])
    return bias_history


def render_colorbar():
    fig, ax = plt.subplots(figsize=(6, 0.5))
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    sm = mpl.cm.ScalarMappable(cmap=COLORMAP_NAME, norm=norm)
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.ax.tick_params(axis='x', colors='white')
    return fig


def render_loss_history(experiment_name, color):
    ex = ExperimentWithDense(experiment_name, verbose=False)
    train_losses = list(ex.load_metadata_entry_history('train_loss'))
    df = pd.DataFrame({
        'training step': np.arange(len(train_losses))[::get_recommended_step_increment(ex)],
        'training loss': train_losses[::get_recommended_step_increment(ex)]
    })
    line_chart = alt.Chart(df).mark_line().encode(
        x='training step:Q',
        y='training loss:Q',
        color=alt.value(color)
    )
    point_chart = alt.Chart(df).mark_circle(size=100).encode(
        x='training step:Q',
        y='training loss:Q',
        color=alt.value(color),
        tooltip=['training step', 'training loss']
        )
    final_chart = (line_chart + point_chart).properties(
        width=800,
        height=300
    )
    return final_chart


def render_accuracy_history(experiment_name, color):
    ex = ExperimentWithDense(experiment_name, verbose=False)
    try:
        train_losses = list(ex.load_metadata_entry_history('train_acc'))
    except KeyError:
        return None
    df = pd.DataFrame({
        'training step': np.arange(len(train_losses))[::get_recommended_step_increment(ex)],
        'training accuracy': train_losses[::get_recommended_step_increment(ex)]
    })
    line_chart = alt.Chart(df).mark_line().encode(
        x='training step:Q',
        y=alt.Y('training accuracy:Q', scale=alt.Scale(domain=[0, 1.0])),
        color=alt.value(color)
    )
    point_chart = alt.Chart(df).mark_circle(size=100).encode(
        x='training step:Q',
        y=alt.Y('training accuracy:Q', scale=alt.Scale(domain=[0, 1.0])),
        color=alt.value(color),
        tooltip=['training step', 'training accuracy']
        )
    final_chart = (line_chart + point_chart).properties(
        width=800,
        height=300
    )
    return final_chart


def render_weight_bar_plots(experiment_name):
    result = []
    colormap = mpl.colormaps[COLORMAP_NAME]
    colormap_norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    num_plots_to_render = sum([
        1 for layer_idx in range(get_num_layers(experiment_name, 0))
        for output_idx in range(get_num_layer_outputs(experiment_name, 0, layer_idx))
        for input_idx in range(-1, get_num_layer_inputs(experiment_name, 0, layer_idx))
    ])
    progress_bar = st.progress(0.0, f"Rendering parameter history...")
    ex = ExperimentWithDense(experiment_name, verbose=False)
    for layer_idx in range(get_num_layers(experiment_name, 0)):
        for output_idx in range(get_num_layer_outputs(experiment_name, 0, layer_idx)):
            for input_idx in range(-1, get_num_layer_inputs(experiment_name, 0, layer_idx)):
                if input_idx == -1:
                    weight_id = f'layer_{layer_idx}_output_{output_idx}'
                    this_weight_history = load_bias_history(experiment_name, layer_idx, output_idx)
                else:
                    weight_id = f'layer_{layer_idx}_input_{input_idx}_output_{output_idx}'
                    this_weight_history = load_weight_history(experiment_name, layer_idx, input_idx, output_idx)
                colors = [mpl.colors.to_hex(colormap(colormap_norm(w))) for w in this_weight_history]
                df = pd.DataFrame({
                    'training step': range(0, ex.last_step+1, get_recommended_step_increment(ex)),
                    'value': this_weight_history,
                    'custom_color': colors
                })
                fig = px.bar(
                    df,
                    x='training step',
                    y='value',
                    color='custom_color',  # Tell Plotly to color by this column
                    color_discrete_map='identity',  # Key: Use the values in 'custom_color' as the actual colors
                )
                fig.update_yaxes(
                    range=[
                        min(-1.0, min(this_weight_history)),
                        max(1.0, max(this_weight_history)),
                    ]
                )
                fig.update_layout(showlegend=False, height=300)
                result.append((fig, weight_id))
                progress_bar.progress(len(result) / num_plots_to_render, f"Rendering parameter history...")
    progress_bar.empty()
    return result

import plotly.graph_objects as go


def plotly_violin_plot(ex, parameter_history, title):
    colormap = mpl.colormaps[COLORMAP_NAME]
    colormap_norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    step_labels = list(range(0, ex.last_step+1, get_recommended_step_increment(ex)))
    fig = go.Figure()
    for step in range(len(parameter_history)):
        step_weights = parameter_history[step].flatten()
        std = step_weights.std()
        fig.add_trace(go.Violin(
            x=[step] * len(step_weights),
            y=step_weights,
            name=f'Step {step_labels[step]}',
            box_visible=True,
            meanline_visible=True,
            fillcolor=f'rgb({",".join([str(int(c*255)) for c in colormap(colormap_norm(std))[:3]])})',
            opacity=0.6,
            line_color=f'white',
            scalemode='width'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Training Step',
        yaxis_title='Distribution',
        showlegend=False,
        xaxis=dict(
            tickmode='array',
            tickvals=step_labels,
            ticktext=[str(s) for s in step_labels]
        ),
        height=600,
        template='plotly_white',
        violinmode='overlay'  # Ensures violins share the same axis
    )
    fig.update_traces(
        width=1,  # Width of each violin
    )
    return fig

def render_weight_violin_plots(experiment_name):
    result = []
    num_plots_to_render = get_num_layers(experiment_name, 0) * 2
    progress_bar = st.progress(0.0, f"Rendering parameter history...")
    ex = ExperimentWithDense(experiment_name, verbose=False)

    for layer_idx in range(get_num_layers(experiment_name, 0)):
        bias_history = load_bias_history(experiment_name, layer_idx)
        weight_history = load_weight_history(experiment_name, layer_idx)
        fig = plotly_violin_plot(ex, weight_history, 'Weight distribution evolution during training. Color depends on standard deviation')
        result.append((fig, f'layer_{layer_idx}_weights'))
        fig = plotly_violin_plot(ex, bias_history, 'Bias distribution evolution during training. Color depends on standard deviation')
        result.append((fig, f'layer_{layer_idx}_biases'))
        progress_bar.progress(len(result) / num_plots_to_render, f"Rendering parameter history...")
    progress_bar.empty()
    return result


@st.fragment
def left_section():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("← Previous step", use_container_width=True, disabled=(st.session_state.epoch == 0)):
            st.session_state.epoch = max(0, st.session_state.epoch - get_recommended_step_increment(ex))
            st.rerun(scope="fragment")

    with col2:
        epoch = st.slider(
            "training step",
            min_value=0,
            max_value=(ex.last_step // get_recommended_step_increment(ex)) * get_recommended_step_increment(ex),
            value=st.session_state.epoch,
            step=get_recommended_step_increment(ex),
            key="epoch_slider"
        )
        if epoch != st.session_state.epoch:
            st.session_state.epoch = epoch
            st.rerun(scope="fragment")

    with col3:
        if st.button("Next step →", use_container_width=True, disabled=(st.session_state.epoch == ex.last_step)):
            st.session_state.epoch = min(ex.last_step, st.session_state.epoch + get_recommended_step_increment(ex))
            st.rerun(scope="fragment")

    st.subheader('Network visualization')
    network_visualization_component(
        experiment_name, st.session_state.epoch, height=50, gap=50, batch_gap=120, last_batch_scale=6.0,
        scroll_key="main_gallery"
    )


@st.fragment
def right_section():
    st.subheader('Loss history')
    training_loss_chart = render_loss_history(experiment_name, mpl.colors.to_hex(mpl.colormaps[COLORMAP_NAME](1.0)))
    st.altair_chart(training_loss_chart, use_container_width=True)
    training_accuracy_chart = render_accuracy_history(experiment_name, mpl.colors.to_hex(mpl.colormaps[COLORMAP_NAME](1.0)))
    if training_accuracy_chart:
        st.subheader('Accuracy history')
        st.altair_chart(training_accuracy_chart, use_container_width=True)

    st.subheader('Parameter history')

    display_selected_parameter_history_on_path_click_js = f"""
    <script>
        (function() {{
            const selectedWeightId = 'selected_weight_id_main_gallery';

            function updateColorDisplay() {{
                const weightId = localStorage.getItem(selectedWeightId);

                const parentDocument = window.parent.document;
                const allWeightHistoryDivs = parentDocument.querySelectorAll('.weight-history-div-above-bar-plot');
                allWeightHistoryDivs.forEach(targetDiv => {{
                    let currentElement = targetDiv;
                    while (currentElement) {{
                        if (currentElement.classList && currentElement.classList.contains("stVerticalBlock")) {{
                            currentElement.parentElement.style.display = "none";
                            return;
                        }}
                        currentElement = currentElement.parentElement;
                    }}
                }});

                if (weightId) {{
                    const selectedWeightHistoryDiv = parentDocument.querySelector('.weight-history-div-above-bar-plot[weight-id="' + weightId + '"]');
                    if (selectedWeightHistoryDiv) {{
                        let currentElement = selectedWeightHistoryDiv;
                        while (currentElement) {{
                            if (currentElement.classList && currentElement.classList.contains("stVerticalBlock")) {{
                                currentElement.parentElement.style.display = "block";
                                return;
                            }}
                            currentElement = currentElement.parentElement;
                        }}
                    }}
                }}
            }}

            updateColorDisplay();
            setInterval(updateColorDisplay, 100);
        }})();
    </script>
    """

    components.html(display_selected_parameter_history_on_path_click_js, height=0, scrolling=False)

    max_num_images = max([
        get_num_layer_inputs(experiment_name, 0, layer_idx)
        for layer_idx in range(get_num_layers(experiment_name, 0))
    ])
    if max_num_images < NUM_NEURONS_THAT_TRIGGERS_VISUALIZATION_SWITCH:
        for fig, weight_id in render_weight_bar_plots(experiment_name):
            chart_container = st.container()
            with chart_container:
                st.markdown(f'<div class="weight-history-div-above-bar-plot" weight-id="{weight_id}"></div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
    else:
        for fig, weight_id in render_weight_violin_plots(experiment_name):
            chart_container = st.container()
            with chart_container:
                st.markdown(f'<div class="weight-history-div-above-bar-plot" weight-id="{weight_id}"></div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    with st.sidebar:
        registry = ExperimentRegistry()
        options = registry.get_experiment_names()
        if 'chosen_experiment_name' in st.session_state and st.session_state.chosen_experiment_name in options:
            experiment_name = st.selectbox("Select experiment:", options, options.index(st.session_state.chosen_experiment_name))
        else:
            experiment_name = st.selectbox("Select experiment:", options)
        st.session_state.chosen_experiment_name = experiment_name
        st.write("Refresh the page to load the latest experiment")
        with st.container(border=True):
            st.write("Experiment config will probably be displayed here at some point")

    ex = ExperimentWithDense(experiment_name, verbose=False)

    if 'calculated_experiments' not in st.session_state or experiment_name not in st.session_state.calculated_experiments:
        st.session_state.layer_image_paths = [None for _ in range(ex.last_step+1)]
        progress_bar = st.progress(0.0, f"Rendering activations for {experiment_name}...")
        for epoch in range(0, ex.last_step+1, get_recommended_step_increment(ex)):
            load_activation_visualizations(experiment_name, epoch)
            load_weights_state(experiment_name, epoch)
            progress_bar.progress(epoch / ex.last_step, f"Rendering activations for {experiment_name}...")
        if 'calculated_experiments' not in st.session_state:
            st.session_state.calculated_experiments = []
        st.session_state.calculated_experiments.append(experiment_name)
        st.rerun()

    st.set_page_config(layout="wide")
    st.title("Network visualization on data with only two features")

    if 'epoch' not in st.session_state:
        st.session_state.epoch = 0

    col1, col2 = st.columns([2, 1], border=True)

    with col1:
        left_section()

    with col2:
        right_section()
        col1, col2 = st.columns([1, 2], border=False)
        with col1:
            st.write('Parameters\' values and predictions are color-coded')
        with col2:
            fig = render_colorbar()
            st.pyplot(fig, transparent=True, width='stretch')
