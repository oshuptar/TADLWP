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
import plotly.graph_objects as go


from experiment_logger import ExperimentRegistry, Experiment


COLORMAP_NAME = 'managua_r'


CUSTOM_MULTISELECT_COLORS = [
    mpl.colors.to_hex(mpl.colormaps[COLORMAP_NAME](0.75)),
    mpl.colors.to_hex(mpl.colormaps[COLORMAP_NAME](0.25)),
    mpl.colors.to_hex(mpl.colormaps[COLORMAP_NAME](1.0)),
    mpl.colors.to_hex(mpl.colormaps[COLORMAP_NAME](0.0))
]

def display_weights_as_one_line(experiment_name, epoch, num_layers, height, last_batch_scale, x_positions, layer_to_gap, layer_to_num_columns):
    weights = load_weights_state(experiment_name, epoch)
    biases = load_biases_state(experiment_name, epoch)
    svg_content = ''
    for layer_idx in range(num_layers):
        is_next_last = (layer_idx + 1 == num_layers)
        width = height
        next_height = int(height * last_batch_scale) if is_next_last else height

        x1 = x_positions[layer_idx] + (width + 1) * layer_to_num_columns[layer_idx]
        x2 = x_positions[layer_idx + 1]

        y1 = 10 + layer_to_gap[layer_idx]//2 + get_num_layer_inputs(experiment_name, epoch, layer_idx) / layer_to_num_columns[layer_idx] // 2 * height + 0.35 * height
        y2 = -15 + layer_to_gap[layer_idx+1]//2 + get_num_layer_outputs(experiment_name, epoch, layer_idx) / layer_to_num_columns[layer_idx+1] // 2 * next_height + 0.35 * next_height

        stroke_width = 20.0
        stroke_opacity = 1.0
        line_color = mpl.colors.to_hex(mpl.colormaps[COLORMAP_NAME](0.5))

        control_offset = (x2 - x1) * 0.5
        path = f'M {x1} {y1} C {x1 + control_offset} {y1}, {x2 - control_offset} {y2}, {x2} {y2}'
        title = f'{len(weights[layer_idx].flatten())} weights'

        weight_id = f'experiment_{experiment_name}_layer_{layer_idx}_weights'
        svg_content += f'<path class="connection-line-display" d="{path}" default-stroke="{line_color}" stroke="{line_color}" stroke-width="{stroke_width}" default-stroke-opacity="{stroke_opacity}" stroke-opacity="{stroke_opacity}" fill="none" style="cursor: pointer; pointer-events: none;" visualization-weight-id="{weight_id}"/>\n'
        svg_content += f'<path class="connection-line-hitbox" d="{path}" stroke="{line_color}" stroke-width="{stroke_width+6}" fill="none" style="cursor: pointer;" weight-id="{weight_id}"><title>{title}</title></path>\n'

        yb = -30 + layer_to_gap[layer_idx+1]//2 + get_num_layer_outputs(experiment_name, epoch, layer_idx) / layer_to_num_columns[layer_idx+1] * next_height + next_height / 2 + height // 3
        xb = x2 - 40
        stroke_width = 40.0
        stroke_opacity = 1.0

        title = f'{len(biases[layer_idx])} biases'
        path = f'M {xb} {yb} h 40'
        weight_id = f'experiment_{experiment_name}_layer_{layer_idx}_biases'
        svg_content += f'<path class="connection-line-display" d="{path}" default-stroke="{line_color}" stroke="{line_color}" stroke-width="{stroke_width}" default-stroke-opacity="{stroke_opacity}" stroke-opacity="{stroke_opacity}" fill="none" style="cursor: pointer; pointer-events: none;" visualization-weight-id="{weight_id}"/>\n'
        svg_content += f'<path class="connection-line-hitbox" d="{path}" stroke="{line_color}" stroke-width="{stroke_width+6}" fill="none" style="cursor: pointer;" weight-id="{weight_id}"><title>{title}</title></path>\n'

    return svg_content


def network_visualization_component(experiment_name, epoch, height=200, gap=10, batch_gap=20, last_batch_scale=1.0, scroll_key=None):
    num_layers = get_num_layers(experiment_name, epoch)
    layer_to_num_images = [get_num_layer_inputs(experiment_name, epoch, layer_idx) for layer_idx in range(num_layers)]
    layer_to_num_images.append(get_num_layer_outputs(experiment_name, epoch, num_layers-1))
    max_images = max(layer_to_num_images)
    layer_to_num_columns = [max(1, int(np.pow(num_images, 1/2.2))) for num_images in layer_to_num_images[:-1]]
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
    other_layers_height = max_images // max(layer_to_num_columns) * height + (max_images - 1)
    total_height_val = other_layers_height // 2
    layer_to_gap = [
        (total_height_val - layer_to_num_images[layer_idx] // layer_to_num_columns[layer_idx] * height) / (layer_to_num_images[layer_idx] + 1)
        for layer_idx in range(num_layers)
    ]
    layer_to_gap.append(gap)

    svg_content = ''
    svg_content += display_weights_as_one_line(experiment_name, epoch, num_layers, height, last_batch_scale, x_positions, layer_to_gap, layer_to_num_columns)

    activation_color = CUSTOM_MULTISELECT_COLORS[st.session_state.chosen_experiment_names.index(experiment_name)]

    weights = load_weights_state(experiment_name, epoch)
    total_height = 0
    for layer_idx in range(num_layers+1):
        x = x_positions[layer_idx]
        is_last_batch = (layer_idx == num_layers)

        image_height = int(height * last_batch_scale) if is_last_batch else height
        image_width = int(img_width * last_batch_scale) if is_last_batch else img_width

        weight_id = f'experiment_{experiment_name}_layer_{layer_idx}_activations'
        y = 10 + layer_to_gap[layer_idx]//2
        h = (image_height + 1) * (layer_to_num_images[layer_idx] // layer_to_num_columns[layer_idx])
        if h > total_height:
            total_height = h
        w = (image_width + 1) * layer_to_num_columns[layer_idx]

        path = f'M {x} {y} h {w} v {h} h -{w} z'
        svg_content += f'''
        <path
            class="connection-line-display"
            d="{path}"
            default-stroke="{activation_color}"
            stroke="{activation_color}"
            stroke-width="10"
            default-stroke-opacity="1.0"
            stroke-opacity="1.0"
            fill="{activation_color}"
            style="cursor: pointer; pointer-events: none;"
            visualization-weight-id="{weight_id}"/>\n
        '''
        if layer_idx > 0:
            title = f'{weights[layer_idx-1].shape[0]} activations per sample'
        else:
            title = f'{weights[0].shape[1]} activations per sample'
        svg_content += f'''
        <path
            d="{path}"
            stroke="{activation_color}"
            stroke-width="20"
            fill="{activation_color}"
            style="cursor: pointer;"
            weight-id="{weight_id}"
            class="connection-line-hitbox"
        ><title>{title}</title></path>
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
    ex = Experiment(experiment_name, verbose=False)
    ex.step = epoch
    return ex.load_torch_model_sequential('model')


@st.cache_data
def load_weights_state(experiment_name, step):
    model = load_model(experiment_name, step)
    weights_state = []
    for m in range(len(model)):
        if isinstance(model[m], nn.Linear):
            weights_state.append(model[m].weight.detach().numpy())
    return weights_state


@st.cache_data
def load_activations_state(experiment_name, step):
    model = load_model(experiment_name, step)
    activations_state = []
    ex = Experiment(experiment_name, verbose=False)
    ex.step = step
    for m in range(len(model)):
        if (m != len(model)-1 or not isinstance(model[m], nn.Linear)) and model[m].__module__ != 'torch.nn.modules.activation':
            continue
        layer_activations_state = []
        for batch in range(ex.last_batch):
            layer_activations_state.extend(ex.load_npy_array(f'batch_{batch}/layer_{m}_activations.npy'))
        activations_state.append(layer_activations_state)
    return activations_state


@st.cache_data
def load_inputs_state(experiment_name, step):
    inputs_state = []
    ex = Experiment(experiment_name, verbose=False)
    ex.step = step
    for batch in range(ex.last_batch):
        inputs_state.extend(ex.load_npy_array(f'batch_{batch}/layer_{0}_inputs.npy'))
    return inputs_state


@st.cache_data
def load_biases_state(experiment_name, step):
    model = load_model(experiment_name, step)
    biases_state = []
    for m in range(len(model)):
        if isinstance(model[m], nn.Linear):
            biases_state.append(model[m].bias.detach().numpy())
    return biases_state


@st.cache_data
def load_parameter_gradient_magnitude_state(experiment_name, step, is_bias=False):
    model = load_model(experiment_name, step)
    ex = Experiment(experiment_name, verbose=False)
    ex.step = 0
    config = ex.load_metadata_entry('config')
    learning_rate = float(config['learning_rate'])
    momentum = float(config['momentum'])
    ex.step = step
    state = []
    layer_idx = 0
    for m in range(len(model)):
        if isinstance(model[m], nn.Linear):
            batch_states = []
            for batch in range(ex.last_batch):
                batch_states.append(ex.load_npy_array(f'batch_{batch}/layer_{layer_idx}_{"bias" if is_bias else "weight"}_gradient_magnitudes.npy'))
            state.append(np.sum(batch_states) * learning_rate / (1 - momentum))
            layer_idx += 1
    return state


@st.cache_data
def load_parameter_optimization_step_magnitude_state(experiment_name, step, is_bias=False):
    model = load_model(experiment_name, step)
    ex = Experiment(experiment_name, verbose=False)
    ex.step = step
    state = []
    layer_idx = 0
    for m in range(len(model)):
        if isinstance(model[m], nn.Linear):
            batch_states = []
            for batch in range(ex.last_batch+1):
                batch_states.append(ex.load_npy_array(f'batch_{batch}/layer_{layer_idx}_{"bias" if is_bias else "weight"}_optimization_step_length.npy'))
            state.append(np.sum(batch_states))
            layer_idx += 1
    return state


@st.cache_data
def load_effective_parameter_change_magnitude_state(experiment_name, step, is_bias=False):
    model = load_model(experiment_name, step)
    ex = Experiment(experiment_name, verbose=False)
    ex.step = step
    state = []
    layer_idx = 0
    for m in range(len(model)):
        if isinstance(model[m], nn.Linear):
            state.append(float(ex.load_npy_array(f'layer_{layer_idx}_{"bias" if is_bias else "weight"}_optimization_step_length.npy')))
            layer_idx += 1
    return state


def get_recommended_step_increment(experiment):
    if experiment.last_step < 20:
        return 1
    return experiment.last_step // 10


def load_weight_history(experiment_name, layer_idx, input_idx=None, output_idx=None):
    ex = Experiment(experiment_name, verbose=False)
    weight_history = []
    for epoch in range(0, ex.last_step+1, get_recommended_step_increment(ex)):
        weights_state = load_weights_state(experiment_name, epoch)
        if input_idx is None:
            weight_history.append(weights_state[layer_idx])
        else:
            weight_history.append(weights_state[layer_idx][output_idx][input_idx])
    return weight_history


def load_activation_history(experiment_name, layer_idx, max_samples=100):
    ex = Experiment(experiment_name, verbose=False)
    activation_history = []
    for epoch in range(0, ex.last_step+1, get_recommended_step_increment(ex)):
        activations_state = load_activations_state(experiment_name, epoch)
        samples = np.asarray(activations_state[layer_idx])
        np.random.shuffle(samples)
        activation_history.append(samples[:max_samples].flatten())
    return activation_history


def load_input_history(experiment_name):
    ex = Experiment(experiment_name, verbose=False)
    input_history = []
    for epoch in range(0, ex.last_step+1, get_recommended_step_increment(ex)):
        inputs_state = load_inputs_state(experiment_name, epoch)
        input_history.append(np.asarray(inputs_state).flatten())
    return input_history


def load_bias_history(experiment_name, layer_idx, output_idx=None):
    ex = Experiment(experiment_name, verbose=False)
    bias_history = []
    for epoch in range(0, ex.last_step+1, get_recommended_step_increment(ex)):
        biases_state = load_biases_state(experiment_name, epoch)
        if output_idx is None:
            bias_history.append(biases_state[layer_idx])
        else:
            bias_history.append(biases_state[layer_idx][output_idx])
    return bias_history


def load_parameter_gradient_magnitude_history(experiment_name, layer_idx, is_bias=False):
    ex = Experiment(experiment_name, verbose=False)
    history = []
    for epoch in range(0, ex.last_step+1, get_recommended_step_increment(ex)):
        state = load_parameter_gradient_magnitude_state(experiment_name, epoch, is_bias)
        history.append(state[layer_idx])
    return history


def load_parameter_optimization_step_magnitude_history(experiment_name, layer_idx, is_bias=False):
    ex = Experiment(experiment_name, verbose=False)
    history = []
    for epoch in range(0, ex.last_step+1, get_recommended_step_increment(ex)):
        state = load_parameter_optimization_step_magnitude_state(experiment_name, epoch, is_bias)
        history.append(state[layer_idx])
    return history


def load_effective_parameter_change_magnitude_history(experiment_name, layer_idx, is_bias=False):
    ex = Experiment(experiment_name, verbose=False)
    history = []
    for epoch in range(0, ex.last_step+1, get_recommended_step_increment(ex)):
        state = load_effective_parameter_change_magnitude_state(experiment_name, epoch, is_bias)
        history.append(state[layer_idx])
    return history


def render_colorbar():
    fig, ax = plt.subplots(figsize=(6, 0.5))
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    sm = mpl.cm.ScalarMappable(cmap=COLORMAP_NAME, norm=norm)
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.ax.tick_params(axis='x', colors='white')
    return fig


def render_metric_history(experiment_names, metric_name):
    data = {
        'epoch': [],
        metric_name: [],
        'experiment': []
    }
    for experiment_name in experiment_names:
        ex = Experiment(experiment_name, verbose=False)
        train_losses = list(ex.load_metadata_entry_history(metric_name))
        values = train_losses[::get_recommended_step_increment(ex)]
        data[metric_name].extend(values)
        data['experiment'].extend([experiment_name for _ in range(len(values))])
        data['epoch'].extend(np.arange(len(train_losses))[::get_recommended_step_increment(ex)])
    df = pd.DataFrame(data)
    color = alt.Color(
        'experiment:N', 
        scale=alt.Scale(
            domain=[experiment_name for experiment_name in experiment_names],
            range=CUSTOM_MULTISELECT_COLORS
        )
    )
    line_chart = alt.Chart(df).mark_line().encode(
        x='epoch:Q',
        y=f'{metric_name}:Q',
        color=color
    )
    point_chart = alt.Chart(df).mark_circle(size=100).encode(
        x='epoch:Q',
        y=f'{metric_name}:Q',
        color=color
    )
    final_chart = (line_chart + point_chart).properties(
        width=800,
        height=400
    )
    return final_chart


def render_optimization_history(experiment_name, layer_idx, is_bias=False):
    ex = Experiment(experiment_name, verbose=False)
    data = {
        'epoch': [],
        'mean parameter change': [],
        'type': []
    }
    parameter_gradient_magnitude_history = load_parameter_gradient_magnitude_history(experiment_name, layer_idx, is_bias)
    parameter_optimization_step_magnitude_history = load_parameter_optimization_step_magnitude_history(experiment_name, layer_idx, is_bias)
    effective_parameter_change_magnitude_history = load_effective_parameter_change_magnitude_history(experiment_name, layer_idx, is_bias)
    data['epoch'].extend(np.arange(ex.last_step + 1)[::get_recommended_step_increment(ex)])
    data['mean parameter change'].extend(parameter_gradient_magnitude_history)
    data['type'].extend(['gradient' for _ in range(len(parameter_gradient_magnitude_history))])
    data['epoch'].extend(np.arange(ex.last_step + 1)[::get_recommended_step_increment(ex)])
    data['mean parameter change'].extend(parameter_optimization_step_magnitude_history)
    data['type'].extend(['optimizer' for _ in range(len(parameter_optimization_step_magnitude_history))])
    data['epoch'].extend(np.arange(ex.last_step + 1)[::get_recommended_step_increment(ex)])
    data['mean parameter change'].extend(effective_parameter_change_magnitude_history)
    data['type'].extend(['effective change' for _ in range(len(effective_parameter_change_magnitude_history))])

    df = pd.DataFrame(data)
    color = CUSTOM_MULTISELECT_COLORS[st.session_state.chosen_experiment_names.index(experiment_name)]

    strokeDash = alt.StrokeDash(
        'type:N',
        scale=alt.Scale(
            domain=['optimizer', 'gradient', 'effective change'],
            range=[[1,0], [5,2], [2,10]]
        )
    )
    line_chart = alt.Chart(df).mark_line(color=color).encode(
        x='epoch:Q',
        y=f'mean parameter change:Q',
        strokeDash=strokeDash
    )
    point_chart = alt.Chart(df).mark_circle(size=100, color=color).encode(
        x='epoch:Q',
        y=f'mean parameter change:Q'
    )
    final_chart = (line_chart + point_chart).properties(
        width=800,
        height=400
    )
    return final_chart


VIOLIN_PLOT_MAX_SAMPLES = 1000


def plotly_violin_plot(ex, parameter_history, title):
    step_labels = list(range(0, ex.last_step+1, get_recommended_step_increment(ex)))
    fig = go.Figure()
    for step in range(len(parameter_history)):
        step_weights = parameter_history[step].flatten()
        np.random.shuffle(step_weights)
        step_weights = step_weights[:VIOLIN_PLOT_MAX_SAMPLES]
        std = step_weights.std()
        fig.add_trace(go.Violin(
            x=[step_labels[step]] * len(step_weights),
            y=step_weights,
            name=f'Step {step_labels[step]}',
            box_visible=True,
            meanline_visible=True,
            fillcolor=CUSTOM_MULTISELECT_COLORS[st.session_state.chosen_experiment_names.index(ex.id)],
            opacity=0.6,
            line_color=f'white',
            scalemode='width'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='epoch',
        yaxis_title='Distribution',
        showlegend=False,
        xaxis=dict(
            tickmode='array',
            tickvals=step_labels,
            ticktext=[str(s) for s in step_labels]
        ),
        height=500,
        template='plotly_white',
        violinmode='overlay'  # Ensures violins share the same axis
    )
    fig.update_traces(
        width=1,  # Width of each violin
    )
    return fig


def render_weight_violin_plots(experiment_name):
    result = []
    num_plots_to_render = get_num_layers(experiment_name, 0) * 3 + 1
    progress_bar = st.progress(0.0, f"Rendering distribution history...")
    ex = Experiment(experiment_name, verbose=False)

    input_history = load_input_history(experiment_name)
    fig = plotly_violin_plot(ex, input_history, 'Input distribution evolution during training')
    result.append((fig, f'experiment_{ex.id}_layer_{0}_activations'))

    for layer_idx in range(get_num_layers(experiment_name, 0)):
        bias_history = load_bias_history(experiment_name, layer_idx)
        weight_history = load_weight_history(experiment_name, layer_idx)
        activation_history = load_activation_history(experiment_name, layer_idx)
        fig = plotly_violin_plot(ex, weight_history, 'Weight distribution evolution during training')
        result.append((fig, f'experiment_{ex.id}_layer_{layer_idx}_weights'))
        fig = plotly_violin_plot(ex, bias_history, 'Bias distribution evolution during training')
        result.append((fig, f'experiment_{ex.id}_layer_{layer_idx}_biases'))
        fig = plotly_violin_plot(ex, activation_history, 'Activation distribution evolution during training')
        result.append((fig, f'experiment_{ex.id}_layer_{layer_idx+1}_activations'))
        progress_bar.progress(len(result) / num_plots_to_render, f"Rendering distribution history...")

    progress_bar.empty()
    return result


@st.fragment
def left_section():
    height = st.slider('Network graph height', 5, 50, 10, key='network_height_slider')
    for experiment_name in st.session_state.chosen_experiment_names:
        ex = Experiment(experiment_name)
        st.subheader(experiment_name)
        network_visualization_component(
            experiment_name, ex.last_step, height=height, gap=0, batch_gap=120, last_batch_scale=1.0,
            scroll_key=experiment_name
        )


@st.fragment
def right_section():

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Loss", "Accuracy", "F1 score", "Distribution history", 'Optimization'])

    experiment_names = st.session_state.chosen_experiment_names

    custom_multiselect_colors_css = "\n<style>\n"
    for i, multiselect_color in enumerate(CUSTOM_MULTISELECT_COLORS):
        custom_multiselect_colors_css += f"""
            [data-baseweb="tag"]:nth-of-type({i+1}) {{
                background-color: {multiselect_color} !important;
                color: white !important;
            }}
        """
    st.markdown(custom_multiselect_colors_css, unsafe_allow_html=True)
    with tab1:
        st.subheader(f'Train loss')
        metric_chart = render_metric_history(experiment_names, 'train_loss')
        st.altair_chart(metric_chart, use_container_width=True)
        st.subheader(f'Validation loss')
        metric_chart = render_metric_history(experiment_names, 'val_loss')
        st.altair_chart(metric_chart, use_container_width=True)

    with tab2:
        st.subheader(f'Add train accuracy here')
        st.subheader(f'Validation accuracy')
        metric_chart = render_metric_history(experiment_names, 'val_acc')
        st.altair_chart(metric_chart, use_container_width=True)

    with tab3:
        st.subheader(f'Add micro f1 score')
        st.subheader(f'Add macro f1 score')

    for experiment_name in experiment_names:
        display_selected_parameter_history_on_path_click_js = f"""
        <script>
            (function() {{
                const selectedWeightId = 'selected_weight_id_{experiment_name}';

                function updateColorDisplay() {{
                    const weightId = localStorage.getItem(selectedWeightId);

                    const parentDocument = window.parent.document;
                    const allWeightHistoryDivs = parentDocument.querySelectorAll('.weight-history-div-above-bar-plot[experiment-id="{experiment_name}"]');
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
                        const allSelectedWeightHistoryDiv = parentDocument.querySelectorAll('.weight-history-div-above-bar-plot[weight-id="' + weightId + '"][experiment-id="{experiment_name}"]');
                        allSelectedWeightHistoryDiv.forEach(selectedWeightHistoryDiv => {{
                            let currentElement = selectedWeightHistoryDiv;
                            while (currentElement) {{
                                if (currentElement.classList && currentElement.classList.contains("stVerticalBlock")) {{
                                    currentElement.parentElement.style.display = "block";
                                    return;
                                }}
                                currentElement = currentElement.parentElement;
                            }}
                        }});
                    }}
                }}

                updateColorDisplay();
                setInterval(updateColorDisplay, 500);
            }})();
        </script>
        """
        components.html(display_selected_parameter_history_on_path_click_js, height=0, scrolling=False)

    with tab4:
        for experiment_name in experiment_names:
            for fig, weight_id in render_weight_violin_plots(experiment_name):
                chart_container = st.container()
                with chart_container:
                    st.markdown(f'<div class="weight-history-div-above-bar-plot" experiment-id="{experiment_name}" weight-id="{weight_id}"></div>', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.title("Optimization step stability")
        st.write(
            '''
            Do individual optimization steps lead in the same direction? The more the optimizer jumps around,
            the higher the sum of its step lengths will be compared to the actual parameter change.

            1. gradient: summary length of steps if a plain SGD was used
            2. optimization step: summary length of steps taken by the optimizer
            3. effective change: how much the parameters actually changed in one epoch
            '''
        )
        for experiment_name in experiment_names:
            progress_bar = st.progress(0.0, f"Rendering optimization history...")
            for layer_idx in range(get_num_layers(experiment_name, 0)):
                fig = render_optimization_history(experiment_name, layer_idx, False)
                weight_id = f'experiment_{experiment_name}_layer_{layer_idx}_weights'
                chart_container = st.container()
                with chart_container:
                    st.markdown(f'<div class="weight-history-div-above-bar-plot" experiment-id="{experiment_name}" weight-id="{weight_id}"></div>', unsafe_allow_html=True)
                    st.altair_chart(fig, use_container_width=True)
                fig = render_optimization_history(experiment_name, layer_idx, True)
                weight_id = f'experiment_{experiment_name}_layer_{layer_idx}_biases'
                chart_container = st.container()
                with chart_container:
                    st.markdown(f'<div class="weight-history-div-above-bar-plot" experiment-id="{experiment_name}" weight-id="{weight_id}"></div>', unsafe_allow_html=True)
                    st.altair_chart(fig, use_container_width=True)
                progress_bar.progress(layer_idx / get_num_layers(experiment_name, 0), f"Rendering optimization history...")
            progress_bar.empty()


def display_experiment_config(experiment_name):
    experiment = Experiment(experiment_name, verbose=False)
    experiment.step = 0
    header_color = CUSTOM_MULTISELECT_COLORS[st.session_state.chosen_experiment_names.index(experiment_name)]
    custom_css = f"""
        <style>
        .button-like-header-{experiment_name} {{
            /* Background and Shape */
            background-color: {header_color}; /* Dark Brown color */
            color: white; /* White font color */
            border-radius: 5px; /* Adjust for more oval shape */
            padding: 5px 10px; /* Vertical and horizontal padding */
            text-align: center; /* Center the text */
            display: inline-block; /* Makes the background only surround the text */
            font-size: 18px; /* Adjust font size */
            margin-bottom: 0px; /* Add some space below the header */

            /* Remove any hover/click effects (since it's not a real button) */
            cursor: default;
            user-select: none;
        }}
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(f'<div class="button-like-header-{experiment_name}">{experiment_name}</div>', unsafe_allow_html=True)
    config = experiment.load_metadata_entry('config')
    
    with st.container(border=True):
        st.write(config)


if __name__ == "__main__":
    with st.sidebar:
        registry = ExperimentRegistry()
        options = registry.get_experiment_names()
        if 'chosen_experiment_names' in st.session_state and st.session_state.chosen_experiment_names in options:
            experiment_names = st.multiselect(
                "Select experiments:",
                options,
                [name for name in st.session_state.chosen_experiment_names],
                max_selections=4
            )
        else:
            experiment_names = st.multiselect(
                "Select experiments:",
                options,
                [options[0]],
                max_selections=4
            )
        st.session_state.chosen_experiment_names = experiment_names
        st.write("Refresh the page to load the latest experiment")
        for i, experiment_name in enumerate(experiment_names):
            display_experiment_config(experiment_name)

    st.set_page_config(layout="wide")
    st.title("Network visualization")

    if 'epoch' not in st.session_state:
        st.session_state.epoch = 0

    col1, col2 = st.columns([1, 1], border=True)

    with col1:
        left_section()

    with col2:
        right_section()
