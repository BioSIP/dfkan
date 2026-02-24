#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author:       Andrés Ortiz, 2026
# Group:        BioSiP Research Group
# License:      MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------
"""
KANExplainability - Versión Completa con Análisis de Importancia Unificado

Integra todas las funcionalidades de explicabilidad para redes KAN:
1. Extracción de funciones aprendidas
2. Visualización de funciones univariadas
3. Análisis de importancia COMPLETO (original + extendido)
4. Análisis de sensibilidad
5. Visualización de flujo de información
6. Knockout selectivo

@author: andres
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class KANExplainability:
    """
    Sistema completo de explicabilidad para redes KAN
    
    Capacidades de Análisis de Importancia:
    ========================================
    
    MÉTODOS ORIGINALES:
    - compute_function_importance(): Varianza, gradiente básico, activación
    - plot_importance(): Visualización de importancia
    
    MÉTODOS EXTENDIDOS:
    - compute_gradient_importance(): Gradientes reales (L2, L1, max, integrated)
    - compute_weighted_function_importance(): Pesos × funciones
    - knockout_analysis(): Knockout selectivo (function/neuron/connection)
    - comprehensive_importance_analysis(): Análisis completo automatizado
    
    OTRAS CAPACIDADES:
    - Extracción y visualización de funciones
    - Análisis de sensibilidad
    - Visualización de arquitectura
    """
    
    def __init__(self, model, feature_names=None, class_names=None):
        """
        Args:
            model: Modelo DualFlexKAN entrenado
            feature_names: Lista con nombres de features de entrada (opcional)
            class_names: Lista con nombres de clases de salida (opcional)
        """
        self.model = model
        self.n_layers = len(model.layers)
        
        # Nombres de features
        if feature_names is None:
            n_features = model.layers[0].in_features
            self.feature_names = [f"x_{i}" for i in range(n_features)]
        else:
            self.feature_names = feature_names
        
        # Nombres de clases
        if class_names is None:
            n_classes = model.layers[-1].out_features
            self.class_names = [f"Class_{i}" for i in range(n_classes)]
        else:
            self.class_names = class_names
    
    # ========================================================================
    # 1. EXTRACCIÓN DE FUNCIONES APRENDIDAS
    # ========================================================================
    
    def extract_all_functions(self):
        """
        Extrae todas las funciones aprendidas de la red
        
        Returns:
            dict: Diccionario con información de todas las funciones
        """
        functions = {}
        
        for layer_idx, layer in enumerate(self.model.layers):
            layer_info = {
                'input_functions': [],
                'output_functions': [],
                'input_strategy': layer.input_function_strategy,
                'output_strategy': layer.output_function_strategy,
                'input_type': layer.input_activation_type,
                'output_type': layer.output_activation_type
            }
            
            # Funciones de entrada
            if hasattr(layer, 'input_function_params') and \
               layer.input_function_params is not None:
                params = layer.input_function_params.detach()
                layer_info['input_functions'] = self._extract_functions(
                    layer.input_activation_fn,
                    params,
                    layer.input_function_strategy
                )
            
            # Funciones de salida
            if hasattr(layer, 'output_function_params') and \
               layer.output_function_params is not None:
                params = layer.output_function_params.detach()
                layer_info['output_functions'] = self._extract_functions(
                    layer.output_activation_fn,
                    params,
                    layer.output_function_strategy
                )
            
            functions[f'layer_{layer_idx}'] = layer_info
        
        return functions
    
    def _extract_functions(self, activation_fn, params, strategy):
        """Extrae funciones individuales según la estrategia"""
        functions = []
        
        if strategy == "global":
            func_params = params[0]
            functions.append({
                'params': func_params.numpy(),
                'equation': self._get_equation(activation_fn, func_params)
            })
        
        elif strategy in ["per_input", "per_neuron"]:
            for i in range(params.shape[0]):
                func_params = params[i]
                functions.append({
                    'params': func_params.numpy(),
                    'equation': self._get_equation(activation_fn, func_params)
                })
        
        elif strategy == "per_neuron_input":
            for i in range(params.shape[0]):
                for j in range(params.shape[1]):
                    func_params = params[i, j]
                    functions.append({
                        'params': func_params.numpy(),
                        'equation': self._get_equation(activation_fn, func_params)
                    })
        
        return functions
    
    def _get_equation(self, activation_fn, params):
        """Obtiene la ecuación de una función"""
        try:
            if hasattr(activation_fn, 'get_equation'):
                return activation_fn.get_equation(params)
            else:
                return str(params.numpy())
        except:
            return "N/A"
    
    # ========================================================================
    # 2. VISUALIZACIÓN DE FUNCIONES
    # ========================================================================
    
    def plot_learned_functions(self, layer_idx=0, function_type='input',
                                x_range=(-3, 3), n_points=1000,
                                max_functions=16, save_path=None):
        """
        Visualiza las funciones aprendidas de una capa
        
        Args:
            layer_idx: Índice de la capa (0-based)
            function_type: 'input' o 'output'
            x_range: Rango de visualización
            n_points: Puntos para graficar
            max_functions: Máximo de funciones a mostrar
            save_path: Ruta para guardar la figura
        """
        layer = self.model.layers[layer_idx]
        
        # Seleccionar funciones
        if function_type == 'input':
            if not hasattr(layer, 'input_function_params') or \
               layer.input_function_params is None:
                print(f"Capa {layer_idx} no tiene funciones de entrada aprendibles")
                return
            params = layer.input_function_params.detach()
            activation_fn = layer.input_activation_fn
            strategy = layer.input_function_strategy
            title_prefix = "Funciones de Entrada"
        else:
            if not hasattr(layer, 'output_function_params') or \
               layer.output_function_params is None:
                print(f"Capa {layer_idx} no tiene funciones de salida aprendibles")
                return
            params = layer.output_function_params.detach()
            activation_fn = layer.output_activation_fn
            strategy = layer.output_function_strategy
            title_prefix = "Funciones de Salida"
        
        # Preparar datos
        x = torch.linspace(x_range[0], x_range[1], n_points)
        
        # Determinar número de funciones
        if strategy == "global":
            n_functions = 1
        elif strategy in ["per_input", "per_neuron"]:
            n_functions = min(params.shape[0], max_functions)
        else:
            print(f"Estrategia {strategy} no soportada para visualización")
            return
        
        # Configurar subplot grid
        n_cols = min(4, n_functions)
        n_rows = (n_functions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_functions == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Graficar cada función
        for i in range(n_functions):
            ax = axes[i]
            
            func_params = params[0] if strategy == "global" else params[i]
            
            with torch.no_grad():
                y = activation_fn(x, func_params).numpy()
            
            ax.plot(x.numpy(), y, linewidth=2, color='steelblue')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
            
            # Título con ecuación
            equation = self._get_equation(activation_fn, func_params)
            if len(equation) > 40:
                equation = equation[:40] + "..."
            ax.set_title(f'φ_{i}: {equation}', fontsize=9)
            
            # Estadísticas
            y_min, y_max = y.min(), y.max()
            ax.text(0.02, 0.98, f'Range: [{y_min:.2f}, {y_max:.2f}]',
                   transform=ax.transAxes, fontsize=7,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Ocultar ejes sobrantes
        for i in range(n_functions, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{title_prefix} - Capa {layer_idx}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # 3. ANÁLISIS DE IMPORTANCIA - MÉTODOS ORIGINALES
    # ========================================================================
    
    def compute_function_importance(self, X_sample, method='variance'):
        """
        Calcula la importancia de cada función (método original)
        
        Args:
            X_sample: Muestra de datos (N, n_features)
            method: 'variance', 'gradient', o 'activation'
        
        Returns:
            dict: Importancia por capa y función
        """
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.FloatTensor(X_sample)
        
        importance = {}
        
        if method == 'variance':
            importance = self._compute_variance_importance(X_sample)
        elif method == 'gradient':
            # Ahora usa el método mejorado
            grad_results = self.compute_gradient_importance(X_sample, method='l2')
            # Convertir a formato compatible con API original
            importance = {'layer_0': {'input': grad_results['feature_importance'].tolist()}}
        elif method == 'activation':
            importance = self._compute_activation_importance(X_sample)
        else:
            raise ValueError(f"Método '{method}' no soportado")
        
        return importance
    
    def _compute_variance_importance(self, X):
        """Importancia basada en varianza de salidas de funciones"""
        importance = {}
        
        with torch.no_grad():
            activations = X
            
            for layer_idx, layer in enumerate(self.model.layers):
                layer_importance = {}
                
                if hasattr(layer, 'input_function_params') and \
                   layer.input_function_params is not None:
                    
                    input_activations = []
                    
                    if layer.input_function_strategy == "per_input":
                        for i in range(layer.in_features):
                            x_i = activations[:, i:i+1]
                            params = layer.input_function_params[i]
                            y_i = layer.input_activation_fn(x_i, params)
                            input_activations.append(y_i.var().item())
                    
                    layer_importance['input'] = input_activations
                
                activations = layer(activations)
                importance[f'layer_{layer_idx}'] = layer_importance
        
        return importance
    
    def _compute_activation_importance(self, X):
        """Importancia basada en magnitud de activaciones"""
        importance = {}
        
        with torch.no_grad():
            activations = X
            
            for layer_idx, layer in enumerate(self.model.layers):
                layer_importance = {}
                
                if hasattr(layer, 'input_function_params') and \
                   layer.input_function_params is not None:
                    
                    input_activations = []
                    
                    if layer.input_function_strategy == "per_input":
                        for i in range(layer.in_features):
                            x_i = activations[:, i:i+1]
                            params = layer.input_function_params[i]
                            y_i = layer.input_activation_fn(x_i, params)
                            input_activations.append(y_i.abs().mean().item())
                    
                    layer_importance['input'] = input_activations
                
                activations = layer(activations)
                importance[f'layer_{layer_idx}'] = layer_importance
        
        return importance
    
    def plot_importance(self, importance_results, layer_idx=0, save_path=None):
        """Visualiza importancia de funciones (método original)"""
        
        layer_key = f'layer_{layer_idx}'
        if layer_key not in importance_results:
            print(f"No hay datos de importancia para capa {layer_idx}")
            return
        
        layer_data = importance_results[layer_key]
        
        if 'input' not in layer_data or not layer_data['input']:
            print(f"No hay importancia de funciones de entrada para capa {layer_idx}")
            return
        
        importances = np.array(layer_data['input'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(importances))
        bars = ax.bar(x_pos, importances, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Colorear por magnitud
        if importances.max() > 0:
            norm_imp = importances / importances.max()
            for bar, imp in zip(bars, norm_imp):
                bar.set_color(plt.cm.YlOrRd(imp))
        
        ax.set_xlabel('Función de Entrada', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importancia', fontsize=12, fontweight='bold')
        ax.set_title(f'Importancia de Funciones - Capa {layer_idx}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'φ_{i}' for i in range(len(importances))])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # 4. ANÁLISIS DE IMPORTANCIA - MÉTODOS EXTENDIDOS
    # ========================================================================
    
    def compute_gradient_importance(self, X_sample, target_class=None, 
                                   method='l2', normalize=True):
        """
        Calcula importancia basada en gradientes reales ∂output/∂input
        
        Args:
            X_sample: Muestra de datos (N, n_features) o (n_features,)
            target_class: Índice de clase objetivo (None = todas)
            method: 'l2', 'l1', 'max', 'integrated'
            normalize: Si True, normaliza las importancias
        
        Returns:
            dict: {
                'feature_importance': array de importancia por feature,
                'gradients_raw': gradientes sin procesar,
                'per_sample': importancia por muestra (si N > 1)
            }
        """
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.FloatTensor(X_sample)
        
        # Asegurar que sea 2D
        if X_sample.dim() == 1:
            X_sample = X_sample.unsqueeze(0)
        
        X_sample.requires_grad_(True)
        
        # Forward pass
        output = self.model(X_sample)
        
        # Seleccionar clase objetivo
        if target_class is not None:
            score = output[:, target_class].sum()
        else:
            score = output.sum()
        
        # Backward pass
        self.model.zero_grad() # Limpiar gradientes previos
        score.backward()
        
        # Obtener gradientes
        gradients = X_sample.grad.detach()  # (N, n_features)
        
        # Calcular importancia según método
        if method == 'l2':
            importance = (gradients ** 2).mean(dim=0).cpu().numpy()
        elif method == 'l1':
            importance = gradients.abs().mean(dim=0).cpu().numpy()
        elif method == 'max':
            importance = gradients.abs().max(dim=0)[0].cpu().numpy()
        elif method == 'integrated':
            importance = self._integrated_gradients(X_sample.detach().cpu())
        else:
            raise ValueError(f"Método '{method}' no soportado")
        
        # Normalizar
        if normalize and importance.sum() > 0:
            importance = importance / importance.sum()
        
        # Importancia por muestra
        per_sample_importance = None
        if X_sample.shape[0] > 1:
            if method == 'l2':
                per_sample_importance = (gradients ** 2).cpu().numpy()
            elif method == 'l1':
                per_sample_importance = gradients.abs().cpu().numpy()
        
        return {
            'feature_importance': importance,
            'gradients_raw': gradients.cpu().numpy(),
            'per_sample': per_sample_importance,
            'method': method,
            'target_class': target_class
        }
    
    def _integrated_gradients(self, X_sample, n_steps=50):
        """Implementación simplificada de Integrated Gradients"""
        baseline = torch.zeros_like(X_sample)
        alphas = torch.linspace(0, 1, n_steps)
        
        integrated_grads = torch.zeros(X_sample.shape[1])
        
        for alpha in alphas:
            X_interp = baseline + alpha * (X_sample - baseline)
            X_interp.requires_grad_(True)
            
            output = self.model(X_interp)
            score = output.sum()
            
            score.backward()
            integrated_grads += X_interp.grad.detach().abs().mean(dim=0)
            
            X_interp.grad.zero_()
        
        integrated_grads /= n_steps
        return integrated_grads.numpy()
    
    def compute_weighted_function_importance(self, X_sample, layer_idx=0,
                                            combine_method='multiply'):
        """
        Analiza importancia combinando magnitud de linear_weights y 
        varianza de funciones
        
        Args:
            X_sample: Muestra de datos para evaluar funciones
            layer_idx: Índice de capa a analizar
            combine_method: 'multiply', 'add', 'weighted_sum'
        
        Returns:
            dict con input_importance, neuron_importance, connection_importance, etc.
        """
        # Asegurarse de que la muestra de entrada esté en el dispositivo correcto
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.FloatTensor(X_sample).to(next(self.model.parameters()).device)
        
        layer = self.model.layers[layer_idx]
        
        # 1. Obtener pesos lineales (estarán en el dispositivo del modelo, ej. cuda:0)
        weights = layer.linear_weights.detach().abs()
        
        # 2. Calcular varianza de funciones
        # --- CORRECCIÓN CLAVE: Crear el tensor en el mismo dispositivo que los pesos ---
        function_variances = torch.zeros(layer.in_features, device=weights.device)
        
        if hasattr(layer, 'input_function_params') and \
           layer.input_function_params is not None:
            
            with torch.no_grad():
                for i in range(layer.in_features):
                    x_i = X_sample[:, i:i+1]
                    
                    if layer.input_function_strategy == "per_input":
                        params = layer.input_function_params[i]
                    elif layer.input_function_strategy == "global":
                        params = layer.input_function_params[0]
                    else:
                        continue
                    
                    y_i = layer.input_activation_fn(x_i, params)
                    
                    # --- CORRECCIÓN CLAVE: Asignar el tensor de varianza directamente, sin .item() ---
                    function_variances[i] = y_i.var()
        else:
            # Si no hay funciones, calcular la varianza de la entrada directamente
            function_variances = X_sample.var(dim=0)
        
        # 3. Combinar pesos y funciones (ambos tensores están ahora en la GPU)
        func_var_expanded = function_variances.unsqueeze(1)
        
        if combine_method == 'multiply':
            connection_importance = weights * func_var_expanded
        elif combine_method == 'add':
            weights_norm = weights / (weights.max() + 1e-8)
            func_var_norm = func_var_expanded / (func_var_expanded.max() + 1e-8)
            connection_importance = weights_norm + func_var_norm
        elif combine_method == 'weighted_sum':
            weights_norm = weights / (weights.max() + 1e-8)
            func_var_norm = func_var_expanded / (func_var_expanded.max() + 1e-8)
            connection_importance = 0.7 * weights_norm + 0.3 * func_var_norm
        else:
            raise ValueError(f"Método '{combine_method}' no soportado")
        
        # 4. Calcular importancias agregadas y mover a CPU para NumPy
        input_importance = connection_importance.sum(dim=1).detach().cpu().numpy()
        neuron_importance = connection_importance.sum(dim=0).detach().cpu().numpy()
        
        return {
            'input_importance': input_importance,
            'neuron_importance': neuron_importance,
            'connection_importance': connection_importance.detach().cpu().numpy(),
            'weights': weights.detach().cpu().numpy(),
            'function_variance': function_variances.detach().cpu().numpy(),
            'combine_method': combine_method,
            'layer_idx': layer_idx
        }
    
    
    def plot_connection_importance(self, importance_results, save_path=None):
        """Visualiza matriz de importancia de conexiones"""
        connection_imp = importance_results['connection_importance']
        layer_idx = importance_results['layer_idx']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Heatmap de conexiones
        ax = axes[0]
        im = ax.imshow(connection_imp, aspect='auto', cmap='YlOrRd', 
                      interpolation='nearest')
        ax.set_xlabel('Neurona de Salida', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature de Entrada', fontsize=11, fontweight='bold')
        ax.set_title(f'Importancia de Conexiones - Capa {layer_idx}', 
                    fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Importancia')
        
        # Añadir valores si la matriz es pequeña
        if connection_imp.shape[0] <= 10 and connection_imp.shape[1] <= 10:
            for i in range(connection_imp.shape[0]):
                for j in range(connection_imp.shape[1]):
                    text = ax.text(j, i, f'{connection_imp[i, j]:.2f}',
                                 ha="center", va="center", color="black", 
                                 fontsize=8)
        
        # 2. Importancia por input
        ax = axes[1]
        input_imp = importance_results['input_importance']
        x_pos = np.arange(len(input_imp))
        bars = ax.bar(x_pos, input_imp, color='steelblue', alpha=0.7, edgecolor='black')
        
        norm_imp = input_imp / (input_imp.max() + 1e-8)
        for bar, imp in zip(bars, norm_imp):
            bar.set_color(plt.cm.YlOrRd(imp))
        
        ax.set_xlabel('Feature de Entrada', fontsize=11, fontweight='bold')
        ax.set_ylabel('Importancia Total', fontsize=11, fontweight='bold')
        ax.set_title('Importancia por Feature', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.feature_names[i] if i < len(self.feature_names) 
                           else f'x_{i}' for i in range(len(input_imp))], 
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Importancia por neurona
        ax = axes[2]
        neuron_imp = importance_results['neuron_importance']
        x_pos = np.arange(len(neuron_imp))
        bars = ax.bar(x_pos, neuron_imp, color='coral', alpha=0.7, edgecolor='black')
        
        norm_imp = neuron_imp / (neuron_imp.max() + 1e-8)
        for bar, imp in zip(bars, norm_imp):
            bar.set_color(plt.cm.YlGn(imp))
        
        ax.set_xlabel('Neurona de Salida', fontsize=11, fontweight='bold')
        ax.set_ylabel('Importancia Total', fontsize=11, fontweight='bold')
        ax.set_title('Importancia por Neurona', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()
        
    def knockout_analysis_fast(self, eval_inputs, baseline_outputs, knockout_type='function', layer_idx=0, metric='output_change'):
        """
        Versión acelerada del análisis de knockout que utiliza un lote de evaluación y 
        una salida de referencia pre-calculados.

        Args:
            eval_inputs (torch.Tensor): Lote de entradas ya en el dispositivo correcto.
            baseline_outputs (torch.Tensor): Salida del modelo para eval_inputs, ya en la CPU.
            knockout_type (str): 'function' o 'neuron'. 'connection' es demasiado lento y se omite.
            layer_idx (int): Índice de la capa a analizar.
            metric (str): Métrica para medir el impacto.
        
        Returns:
            dict: Resultados del análisis de knockout.
        """
        if knockout_type == 'connection':
            print("ADVERTENCIA: El knockout a nivel de conexión es computacionalmente inviable y se omitirá.")
            return None

        self.model.eval()
        layer = self.model.layers[layer_idx]
        impacts = []
        elements = []

        if knockout_type == 'function':
            if not hasattr(layer, 'input_function_params') or layer.input_function_params is None:
                print(f"Capa {layer_idx} no tiene funciones de entrada para knockout.")
                return None
            
            n_functions = layer.in_features if layer.input_function_strategy == 'per_input' else 1
            elements = [f'func_{i}' for i in range(n_functions)]
            original_params = layer.input_function_params.data.clone()
            
            for i in range(n_functions):
                with torch.no_grad():
                    # Apaga la función (la convierte en una identidad simple)
                    if layer.input_function_strategy == 'per_input':
                        layer.input_function_params.data[i].fill_(0)
                        layer.input_function_params.data[i, 0] = 1.0
                    
                    output_knockout = self.model(eval_inputs).cpu().numpy()
                
                impacts.append(self._compute_impact(baseline_outputs, output_knockout, metric))
                
                # Restaura los parámetros originales inmediatamente
                layer.input_function_params.data = original_params.clone()
        
        elif knockout_type == 'neuron':
            n_neurons = layer.out_features
            elements = [f'neuron_{i}' for i in range(n_neurons)]
            original_weights = layer.linear_weights.data.clone()
            # Asumimos que el bias existe, si no, habría que añadir un check
            original_bias = layer.bias.data.clone()

            for i in range(n_neurons):
                with torch.no_grad():
                    # Apaga la neurona
                    layer.linear_weights.data[:, i] = 0
                    layer.bias.data[i] = 0
                    
                    output_knockout = self.model(eval_inputs).cpu().numpy()
                
                impacts.append(self._compute_impact(baseline_outputs, output_knockout, metric))

                # Restaura los pesos y bias originales
                layer.linear_weights.data = original_weights.clone()
                layer.bias.data = original_bias.clone()

        return {
            'knockout_type': knockout_type,
            'layer_idx': layer_idx,
            'metric': metric,
            'impacts': np.array(impacts),
            'elements': elements
        }
    
    def knockout_analysis(self, X_sample, knockout_type='function',
                         layer_idx=0, metric='output_change'):
        """
        Analiza impacto de desactivar funciones/conexiones individuales
        """
        # Asegurarse de que X_sample está en el dispositivo correcto
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.FloatTensor(X_sample).to(next(self.model.parameters()).device)
        
        layer = self.model.layers[layer_idx]
        
        # Output de referencia
        with torch.no_grad():
            # --- CORRECCIÓN ---
            output_original = self.model(X_sample).cpu().numpy()
        
        if knockout_type == 'function':
            return self._knockout_functions(X_sample, layer, layer_idx, 
                                          output_original, metric)
        elif knockout_type == 'neuron':
            return self._knockout_neurons(X_sample, layer, layer_idx,
                                        output_original, metric)
        elif knockout_type == 'connection':
            return self._knockout_connections(X_sample, layer, layer_idx,
                                            output_original, metric)
        else:
            raise ValueError(f"Tipo '{knockout_type}' no soportado")
    
    def _knockout_functions(self, X_sample, layer, layer_idx, 
                           output_original, metric):
        """Knockout de funciones individuales"""
        
        if not hasattr(layer, 'input_function_params') or \
           layer.input_function_params is None:
            print(f"Capa {layer_idx} no tiene funciones de entrada para knockout")
            return None
        
        impacts = []
        original_params = layer.input_function_params.data.clone()
        
        n_functions = layer.in_features if layer.input_function_strategy == 'per_input' else 1
        
        for i in range(n_functions):
            with torch.no_grad():
                if layer.input_function_strategy == 'per_input':
                    saved_params = layer.input_function_params[i].clone()
                    layer.input_function_params[i].fill_(0)
                    layer.input_function_params[i, 0] = 1.0  # f(x) = x
                
                # --- CORRECCIÓN ---
                output_knockout = self.model(X_sample).cpu().numpy()
                
                if layer.input_function_strategy == 'per_input':
                    layer.input_function_params[i] = saved_params
            
            impact = self._compute_impact(output_original, output_knockout, metric)
            impacts.append(impact)
        
        layer.input_function_params.data = original_params
        
        return {
            'knockout_type': 'function',
            'layer_idx': layer_idx,
            'metric': metric,
            'impacts': np.array(impacts),
            'elements': [f'func_{i}' for i in range(n_functions)]
        }
    
    def _knockout_neurons(self, X_sample, layer, layer_idx,
                         output_original, metric):
        """Knockout de neuronas individuales"""
        
        impacts = []
        original_bias = layer.bias.data.clone()
        original_weights = layer.linear_weights.data.clone()
        
        for i in range(layer.out_features):
            with torch.no_grad():
                layer.bias[i] = 0
                layer.linear_weights[:, i] = 0
                
                # --- CORRECCIÓN ---
                output_knockout = self.model(X_sample).cpu().numpy()
                
                layer.bias[i] = original_bias[i]
                layer.linear_weights[:, i] = original_weights[:, i]
            
            impact = self._compute_impact(output_original, output_knockout, metric)
            impacts.append(impact)
        
        return {
            'knockout_type': 'neuron',
            'layer_idx': layer_idx,
            'metric': metric,
            'impacts': np.array(impacts),
            'elements': [f'neuron_{i}' for i in range(layer.out_features)]
        }
    
    def _knockout_connections(self, X_sample, layer, layer_idx,
                             output_original, metric):
        """Knockout de conexiones individuales"""
        
        impacts = []
        elements = []
        original_weights = layer.linear_weights.data.clone()
        
        for i in range(layer.in_features):
            for j in range(layer.out_features):
                with torch.no_grad():
                    saved_weight = layer.linear_weights[i, j].item()
                    layer.linear_weights[i, j] = 0
                    
                    # --- CORRECCIÓN ---
                    output_knockout = self.model(X_sample).cpu().numpy()
                    
                    layer.linear_weights[i, j] = saved_weight
                
                impact = self._compute_impact(output_original, output_knockout, metric)
                impacts.append(impact)
                elements.append(f'conn_{i}_{j}')
        
        layer.linear_weights.data = original_weights
        
        return {
            'knockout_type': 'connection',
            'layer_idx': layer_idx,
            'metric': metric,
            'impacts': np.array(impacts),
            'elements': elements
        }
    
    def _compute_impact(self, output_original, output_knockout, metric):
        """Calcula métrica de impacto entre outputs"""
        
        if metric == 'output_change':
            return np.abs(output_original - output_knockout).mean()
        elif metric == 'mse':
            return ((output_original - output_knockout) ** 2).mean()
        elif metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(output_original.reshape(1, -1), 
                                   output_knockout.reshape(1, -1))[0, 0]
            return 1 - sim
        else:
            return np.abs(output_original - output_knockout).mean()
    
    def _knockout_top_k(self, X_sample, baseline_outputs, layer, layer_idx, top_k_indices, metric):
        """
        Método auxiliar que realiza knockout de funciones o neuronas solo para los
        índices especificados (top-k).
        """
        # --- CORRECCIÓN: Verificar si existen parámetros antes de intentar acceder a .data ---
        if not hasattr(layer, 'input_function_params') or layer.input_function_params is None:
            # Si la estrategia es 'none' o 'fixed', no hay parámetros que apagar
            return None
        # -----------------------------------------------------------------------------------

        impacts = []
        elements = []
        original_params = layer.input_function_params.data.clone()

        for i in top_k_indices:
            elements.append(f'func_{i}')
            with torch.no_grad():
                if layer.input_function_strategy == 'per_input':
                    # Apaga la función y la convierte en identidad (o cero, según diseño)
                    # Aquí asumimos que el primer parámetro controla la magnitud o es el peso lineal
                    layer.input_function_params.data[i].fill_(0)
                    
                    # Opcional: Si quieres que sea identidad f(x)=x, depende de tu base.
                    # Si es polinomial, params[0] suele ser el bias o coef de grado 0.
                    # Para anularla completamente (f(x)=0), dejar todo en 0 es correcto.
                    # layer.input_function_params.data[i, 0] = 1.0 
                
                elif layer.input_function_strategy == 'global':
                     layer.input_function_params.data[0].fill_(0)

                output_knockout = self.model(X_sample).cpu().numpy()
            
            impact = self._compute_impact(baseline_outputs, output_knockout, metric)
            impacts.append(impact)
            
            # Restaura todos los parámetros en cada iteración para aislar el efecto
            layer.input_function_params.data = original_params.clone()
        
        return {
            'knockout_type': 'function (Top-K)',
            'layer_idx': layer_idx,
            'metric': metric,
            'impacts': np.array(impacts),
            'elements': elements
        }
    
    def plot_knockout_analysis(self, knockout_results, top_k=None, save_path=None):
        """Visualiza resultados del análisis de knockout"""
        
        impacts = knockout_results['impacts']
        elements = knockout_results['elements']
        knockout_type = knockout_results['knockout_type']
        
        sorted_indices = np.argsort(impacts)[::-1]
        
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
        
        impacts_sorted = impacts[sorted_indices]
        elements_sorted = [elements[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(impacts_sorted) * 0.3)))
        
        y_pos = np.arange(len(impacts_sorted))
        bars = ax.barh(y_pos, impacts_sorted, color='crimson', alpha=0.7, 
                      edgecolor='black')
        
        norm_impacts = impacts_sorted / (impacts_sorted.max() + 1e-8)
        for bar, imp in zip(bars, norm_impacts):
            bar.set_color(plt.cm.Reds(0.3 + 0.7 * imp))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(elements_sorted)
        ax.invert_yaxis()
        ax.set_xlabel('Impacto (cambio en output)', fontsize=12, fontweight='bold')
        ax.set_title(f'Knockout Analysis: {knockout_type.capitalize()}s - ' + 
                    f'Capa {knockout_results["layer_idx"]}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, impact) in enumerate(zip(bars, impacts_sorted)):
            ax.text(impact, i, f' {impact:.4f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # 5. ANÁLISIS COMPLETO DE IMPORTANCIA
    # ========================================================================
    
    def comprehensive_importance_analysis(self, X_sample, layer_idx=0, 
                                         save_dir='./importance_analysis', 
                                         top_k_knockout=20,
                                         enable_knockout=True): 
        """
        Realiza análisis completo de importancia. 
        
        Args:
            X_sample: Muestra de datos.
            layer_idx: Capa a analizar.
            save_dir: Directorio para guardar resultados.
            top_k_knockout: Número de funciones más importantes a analizar con knockout.
            enable_knockout: (bool) Si es False, salta el análisis de knockout (más rápido).
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("="*70)
        print("ANÁLISIS COMPLETO DE IMPORTANCIA - KAN")
        print(f"Modo Knockout: {'ACTIVADO (Top-K)' if enable_knockout else 'DESACTIVADO'}")
        print("="*70)
        
        device = next(self.model.parameters()).device
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.FloatTensor(X_sample)
        X_sample = X_sample.to(device)
        
        # Inicializar variables de retorno
        knockout_func = None
        knockout_neur = None

        # 1. Gradientes reales (RÁPIDO)
        print(f"\n1. Calculando importancia por gradientes...")
        grad_imp = self.compute_gradient_importance(X_sample, method='l2')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(grad_imp['feature_importance']))
        ax.bar(x_pos, grad_imp['feature_importance'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importancia (L2 norm gradientes)', fontsize=12, fontweight='bold')
        ax.set_title('Importancia por Gradientes', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gradient_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Guardado en {save_dir}/gradient_importance.png")
        
        # 2. Pesos + Funciones (RÁPIDO)
        print(f"\n2. Analizando pesos lineales + funciones...")
        weighted_imp = self.compute_weighted_function_importance(X_sample, layer_idx=layer_idx)
        self.plot_connection_importance(weighted_imp, save_path=f'{save_dir}/connection_importance.png')
        print(f"   ✓ Guardado en {save_dir}/connection_importance.png")
        
        # --- LÓGICA CONDICIONAL PARA KNOCKOUT ---
        if enable_knockout:
            print("\nCalculando salida de referencia para análisis de knockout...")
            with torch.no_grad():
                baseline_outputs = self.model(X_sample).cpu().numpy()
                
            # 3. Knockout de las funciones Top-K
            print(f"\n3. Knockout de las Top {top_k_knockout} funciones más importantes...")
            
            # Obtenemos los índices de las funciones más importantes del análisis de gradientes
            top_k_indices = np.argsort(grad_imp['feature_importance'])[::-1][:top_k_knockout]
            
            knockout_func = self._knockout_top_k(X_sample, baseline_outputs, 
                                                 self.model.layers[layer_idx], layer_idx,
                                                 top_k_indices, metric='output_change')
            if knockout_func is not None:
                self.plot_knockout_analysis(knockout_func, save_path=f'{save_dir}/knockout_top_k_functions.png')
                print(f"   ✓ Guardado en {save_dir}/knockout_top_k_functions.png")

            # 4. Knockout de neuronas
            print(f"\n4. Knockout de neuronas (acelerado)...")
            knockout_neur = self.knockout_analysis_fast(X_sample, baseline_outputs,
                                                       knockout_type='neuron',
                                                       layer_idx=layer_idx, metric='output_change')
            if knockout_neur is not None:
                self.plot_knockout_analysis(knockout_neur, save_path=f'{save_dir}/knockout_neurons.png')
                print(f"   ✓ Guardado en {save_dir}/knockout_neurons.png")
        else:
            print("\n3. & 4. Análisis de Knockout omitido por configuración.")

        # Resumen
        print("\n" + "="*70)
        print("RESUMEN DE RESULTADOS")
        print("="*70)
        top_features_indices = np.argsort(grad_imp['feature_importance'])[::-1][:3]
        print(f"\nTop 3 Features más importantes (por gradientes):")
        for i, idx in enumerate(top_features_indices, 1):
            fname = self.feature_names[idx] if idx < len(self.feature_names) else f'x_{idx}'
            print(f"  {i}. {fname} (índice {idx}): {grad_imp['feature_importance'][idx]:.4f}")
        
        if knockout_func is not None:
            print(f"\nImpacto de knockout en las funciones más críticas:")
            for i in range(min(3, len(knockout_func['elements']))):
                print(f"  {i+1}. {knockout_func['elements'][i]}: {knockout_func['impacts'][i]:.4f}")

        if knockout_neur is not None:
            print(f"\nTop 3 Neuronas más importantes (por knockout):")
            top_neurons = np.argsort(knockout_neur['impacts'])[::-1][:3]
            for i, idx in enumerate(top_neurons, 1):
                print(f"  {i}. {knockout_neur['elements'][idx]}: {knockout_neur['impacts'][idx]:.4f}")
        
        print("\n" + "="*70)
        print(f"Análisis completo guardado en: {save_dir}")
        print("="*70)
        
        return {
            'gradient_importance': grad_imp,
            'weighted_importance': weighted_imp,
            'knockout_functions': knockout_func,
            'knockout_neurons': knockout_neur
        }
    
    # ========================================================================
    # 6. ANÁLISIS DE SENSIBILIDAD (MÉTODO ORIGINAL)
    # ========================================================================
    
    def sensitivity_analysis(self, X_sample, feature_idx=0, 
                            perturbation_range=(-2, 2), n_steps=50):
        """
        Analiza cómo cambia la salida al perturbar una feature
        
        Args:
            X_sample: Punto de referencia (1, n_features)
            feature_idx: Índice de la feature a perturbar
            perturbation_range: Rango de perturbación
            n_steps: Número de pasos
        
        Returns:
            dict: Resultados del análisis
        """
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.FloatTensor(X_sample)
        
        if X_sample.dim() == 1:
            X_sample = X_sample.unsqueeze(0)
        
        perturbations = np.linspace(perturbation_range[0], 
                                    perturbation_range[1], 
                                    n_steps)
        
        outputs = []
        
        with torch.no_grad():
            for perturb in perturbations:
                X_perturbed = X_sample.clone()
                X_perturbed[0, feature_idx] += perturb
                output = self.model(X_perturbed)
                outputs.append(output[0].numpy())
        
        outputs = np.array(outputs)
        
        return {
            'perturbations': perturbations,
            'outputs': outputs,
            'feature_idx': feature_idx,
            'feature_name': self.feature_names[feature_idx],
            'original_value': X_sample[0, feature_idx].item()
        }
    
    def plot_sensitivity(self, sensitivity_results, save_path=None):
        """Visualiza análisis de sensibilidad"""
        
        perturbations = sensitivity_results['perturbations']
        outputs = sensitivity_results['outputs']
        feature_name = sensitivity_results['feature_name']
        original_value = sensitivity_results['original_value']
        
        n_classes = outputs.shape[1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for class_idx in range(n_classes):
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f'Output_{class_idx}'
            ax.plot(perturbations + original_value, outputs[:, class_idx],
                   label=class_name, linewidth=2)
        
        ax.axvline(original_value, color='red', linestyle='--', 
                  label='Valor Original', linewidth=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f'{feature_name} (perturbado)', fontsize=12)
        ax.set_ylabel('Probabilidad / Activación', fontsize=12)
        ax.set_title(f'Análisis de Sensibilidad: {feature_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # 7. VISUALIZACIÓN DE ARQUITECTURA (MÉTODO ORIGINAL)
    # ========================================================================
    
    def visualize_network_architecture(self, save_path=None):
        """Visualiza la arquitectura completa de la red"""
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        layer_info = []
        for layer in self.model.layers:
            layer_info.append({
                'in': layer.in_features,
                'out': layer.out_features,
                'input_strategy': layer.input_function_strategy,
                'output_strategy': layer.output_function_strategy
            })
        
        n_layers = len(layer_info) + 1
        x_positions = np.linspace(1, 9, n_layers)
        
        for i, x_pos in enumerate(x_positions):
            if i == 0:
                n_nodes = layer_info[0]['in']
                label = f"Input\n({n_nodes})"
            elif i < len(x_positions) - 1:
                n_nodes = layer_info[i-1]['out']
                label = f"L{i}\n({n_nodes})"
            else:
                n_nodes = layer_info[-1]['out']
                label = f"Output\n({n_nodes})"
            
            y_pos = 5
            height = min(3, n_nodes * 0.3)
            width = 0.5
            
            rect = FancyBboxPatch(
                (x_pos - width/2, y_pos - height/2),
                width, height,
                boxstyle="round,pad=0.05",
                edgecolor='black',
                facecolor='lightblue' if i == 0 else ('lightgreen' if i == len(x_positions)-1 else 'lightyellow'),
                linewidth=2
            )
            ax.add_patch(rect)
            
            ax.text(x_pos, y_pos, label, ha='center', va='center',
                   fontsize=10, fontweight='bold')
            
            if 0 < i < len(x_positions):
                strategy_text = f"{layer_info[i-1]['input_strategy'][:8]}\n{layer_info[i-1]['output_strategy'][:8]}"
                ax.text(x_pos, y_pos - height/2 - 0.5, strategy_text,
                       ha='center', va='top', fontsize=7, style='italic')
        
        for i in range(len(x_positions) - 1):
            arrow = FancyArrowPatch(
                (x_positions[i] + 0.25, 5),
                (x_positions[i+1] - 0.25, 5),
                arrowstyle='->', mutation_scale=20,
                linewidth=2, color='gray', alpha=0.6
            )
            ax.add_patch(arrow)
        
        plt.title('Arquitectura de Red KAN', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # 8. GENERACIÓN DE INFORME COMPLETO
    # ========================================================================
    
    def generate_explainability_report(self, X_sample, output_dir='./explainability_report'):
        """
        Genera un reporte completo de explicabilidad
        
        Args:
            X_sample: Muestra de datos para análisis
            output_dir: Directorio para guardar archivos
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*60)
        print("GENERANDO REPORTE DE EXPLICABILIDAD")
        print("="*60)
        
        # 1. Arquitectura
        print("\n1. Visualizando arquitectura...")
        self.visualize_network_architecture(
            save_path=f'{output_dir}/architecture.png'
        )
        
        # 2. Funciones aprendidas
        print("\n2. Extrayendo funciones aprendidas...")
        functions = self.extract_all_functions()
        
        print("\n3. Visualizando funciones por capa...")
        for layer_idx in range(self.n_layers):
            self.plot_learned_functions(
                layer_idx=layer_idx,
                function_type='input',
                save_path=f'{output_dir}/functions_layer_{layer_idx}_input.png'
            )
        
        # 3. Análisis de importancia completo
        print("\n4. Análisis completo de importancia...")
        importance_results = self.comprehensive_importance_analysis(
            X_sample,
            layer_idx=0,
            save_dir=f'{output_dir}/importance'
        )
        
        # 4. Sensibilidad (primera feature)
        print("\n5. Análisis de sensibilidad...")
        sensitivity = self.sensitivity_analysis(X_sample[0:1], feature_idx=0)
        self.plot_sensitivity(
            sensitivity,
            save_path=f'{output_dir}/sensitivity_feature_0.png'
        )
        
        print("\n" + "="*60)
        print(f"REPORTE COMPLETO GUARDADO EN: {output_dir}")
        print("="*60)


# ============================================================================
# FUNCIÓN DE CONVENIENCIA
# ============================================================================

def quick_explain(model, X_sample, feature_names=None, class_names=None):
    """
    Análisis rápido de explicabilidad
    
    Args:
        model: Modelo KAN entrenado
        X_sample: Muestra de datos
        feature_names: Nombres de features (opcional)
        class_names: Nombres de clases (opcional)
    """
    explainer = KANExplainability(model, feature_names, class_names)
    
    print("\n Funciones de Entrada - Capa 1:")
    explainer.plot_learned_functions(layer_idx=0, function_type='input')
    
    print("\nAnálisis Completo de Importancia:")
    results = explainer.comprehensive_importance_analysis(X_sample, layer_idx=0)
    
    return explainer, results



###################### VISUALIZACION ########################
def plot_aggregated_importance_heatmap(feature_importance, feature_names, image_shape=(28, 28)):
    """
    Visualiza la importancia agregada de las características como un mapa de calor.
    """
    if len(feature_importance) != np.prod(image_shape):
        print("Error: La longitud del vector de importancia no coincide con la forma de la imagen.")
        # Para datos tabulares, usar un gráfico de barras
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.xticks(range(len(feature_importance)), feature_names, rotation=90)
        plt.title('Importancia Agregada por Característica')
        plt.ylabel('Importancia')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        return

    # Remodelar el vector de importancia a la forma de la imagen
    importance_map = feature_importance.reshape(image_shape)

    plt.figure(figsize=(8, 6))
    plt.imshow(importance_map, cmap='inferno') # 'inferno' o 'hot' funcionan bien
    plt.colorbar(label='Importancia Agregada')
    plt.title('Mapa de Calor de Importancia de Características (Agregado)')
    plt.xlabel('Píxel (X)')
    plt.ylabel('Píxel (Y)')
    plt.show()

# --- Uso ---
# feature_importance = grad_results['feature_importance']
# feature_names = [f'pixel_{i}' for i in range(784)] # Nombres de ejemplo
# plot_aggregated_importance_heatmap(feature_importance, feature_names)



def plot_saliency_map(original_image, gradients_raw_sample, image_shape=(28, 28)):
    """
    Visualiza la imagen original junto a su mapa de saliencia.
    """
    saliency_map = gradients_raw_sample.reshape(image_shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Imagen Original
    axes[0].imshow(original_image.reshape(image_shape), cmap='gray')
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Mapa de Saliencia
    # Usamos un colormap divergente (RdBu_r) para mostrar gradientes positivos (azul) y negativos (rojo)
    im = axes[1].imshow(saliency_map, cmap='RdBu_r')
    axes[1].set_title('Mapa de Saliencia (Gradientes)')
    axes[1].axis('off')
    
    fig.colorbar(im, ax=axes[1], label='Valor del Gradiente')
    plt.show()

# --- Uso ---
# sample_idx = 0  # Elige la muestra que quieres explicar
# X_sample, _ = next(iter(test_loader)) # Carga un batch
# original_image = X_sample[sample_idx].cpu().numpy()
# gradients_raw_sample = gradients_raw[sample_idx]
#
# plot_saliency_map(original_image, gradients_raw_sample)



def plot_importance_distribution(per_sample_importance, feature_names, top_k=10):
    """
    Muestra la distribución de la importancia para las top_k características más importantes.
    """
    # Calcular la importancia media para encontrar las top_k
    mean_importance = per_sample_importance.mean(axis=0)
    top_indices = np.argsort(mean_importance)[-top_k:]
    
    # Filtrar los datos y nombres para las top_k características
    top_k_data = per_sample_importance[:, top_indices]
    top_k_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(12, 7))
    plt.boxplot(top_k_data, labels=top_k_names, vert=False)
    plt.title(f'Distribución de Importancia para las {top_k} Características Principales')
    plt.xlabel('Importancia (Magnitud)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# --- Uso ---
# per_sample_importance = grad_results['per_sample']
# feature_names = [f'pixel_{i}' for i in range(784)]
# plot_importance_distribution(per_sample_importance, feature_names, top_k=15)

