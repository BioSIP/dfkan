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
Created on Fri Aug  1 11:01:37 2025

@author: andres
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.special import eval_gegenbauer, eval_jacobi
import math
from .KAN_activation import *


class DualFlexKANLinear(nn.Module):
    """
    Capa KAN con control independiente de funciones de entrada y activación de salida
    y regularización flexible (dropout y batch norm) con control de orden de aplicación
    
    Permite combinar:
    - Funciones de entrada: transforman cada entrada antes de la combinación lineal
    - Funciones de activación: transforman la salida de cada neurona
    - Regularización flexible: dropout y batch norm en diferentes posiciones
    
    Estrategias para funciones de entrada:
    - "none": Sin transformación de entrada
    - "fixed": Función fija (se especifica en input_activation_type)
    - "global": Una función compartida para todas las entradas
    - "per_input": Una función por cada característica de entrada
    - "per_neuron_input": Una función por cada entrada de cada neurona
    
    Estrategias para funciones de activación:
    - "none": Sin activación (salida lineal)
    - "fixed": Función fija (se especifica en output_activation_type)
    - "global": Una función compartida para todas las neuronas
    - "per_neuron": Una función por cada neurona
    
    Regularización flexible:
    - dropout_position: "before_activation", "after_activation", "both", "none"
    - batch_norm_position: "before_activation", "after_activation", "both", "none"
    - regularization_order: "dropout_first", "batch_norm_first" (cuando ambos están en la misma posición)
    """
    
    def __init__(self, in_features, out_features, 
                 # Configuración funciones de entrada
                 input_function_strategy="per_input",
                 input_activation_type="polynomial",
                 input_activation_kwargs=None,
                 # Configuración funciones de activación
                 output_function_strategy="per_neuron", 
                 output_activation_type="polynomial",
                 output_activation_kwargs=None,
                 # Regularización flexible
                 dropout_prob=0.0,
                 dropout_position="after_activation",  # "before_activation", "after_activation", "both", "none"
                 use_batch_norm=False,
                 batch_norm_position="after_activation",  # "before_activation", "after_activation", "both", "none"
                 regularization_order="dropout_first",  # "dropout_first", "batch_norm_first"
                 batch_norm_momentum=0.1,
                 batch_norm_eps=1e-5
                 ):
        
        super(DualFlexKANLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.input_function_strategy = input_function_strategy
        self.output_function_strategy = output_function_strategy
        self.input_activation_type = input_activation_type
        self.output_activation_type = output_activation_type
        self.dropout_prob = dropout_prob
        self.dropout_position = dropout_position
        self.use_batch_norm = use_batch_norm
        self.batch_norm_position = batch_norm_position
        self.regularization_order = regularization_order
        
        # Valores por defecto para kwargs
        if input_activation_kwargs is None:
            input_activation_kwargs = {}
        if output_activation_kwargs is None:
            output_activation_kwargs = {}
            
        # Configurar funciones de entrada
        self._setup_input_functions(input_activation_kwargs)
        
        # Configurar funciones de activación de salida  
        self._setup_output_functions(output_activation_kwargs)
        
        # Pesos lineales siempre presentes
        self.linear_weights = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Configurar regularización flexible
        self._setup_flexible_regularization(batch_norm_momentum, batch_norm_eps)
    
    def _setup_flexible_regularization(self, momentum, eps):
        """Configura dropout y batch norm según las posiciones especificadas"""
        
        # Dropout antes de la activación
        if self.dropout_prob > 0 and self.dropout_position in ["before_activation", "both"]:
            self.dropout_before = nn.Dropout(self.dropout_prob)
        else:
            self.dropout_before = None
            
        # Dropout después de la activación
        if self.dropout_prob > 0 and self.dropout_position in ["after_activation", "both"]:
            self.dropout_after = nn.Dropout(self.dropout_prob)
        else:
            self.dropout_after = None
            
        # Batch norm antes de la activación
        if self.use_batch_norm and self.batch_norm_position in ["before_activation", "both"]:
            self.batch_norm_before = nn.BatchNorm1d(self.out_features, momentum=momentum, eps=eps)
        else:
            self.batch_norm_before = None
            
        # Batch norm después de la activación
        if self.use_batch_norm and self.batch_norm_position in ["after_activation", "both"]:
            self.batch_norm_after = nn.BatchNorm1d(self.out_features, momentum=momentum, eps=eps)
        else:
            self.batch_norm_after = None
    
    def _create_activation_function(self, activation_type, **kwargs):
        """Crea función de activación según el tipo"""
        if activation_type == "polynomial":
            return PolynomialActivation(**kwargs)
        elif activation_type == "legendre":
            return OptimizedLegendreActivation(**kwargs)
        elif activation_type == "chebyshev":
            return ChebyshevActivation(**kwargs)
        elif activation_type == "gegenbauer":
            return GegenbauerActivation(**kwargs)
        elif activation_type == "jacobi":
            return JacobiActivation(**kwargs)
        elif activation_type == "bspline":
            return BSplineActivation(**kwargs)
        elif activation_type == "fbspline" or activation_type == "fast_bspline":
            return FastBSplineActivation(**kwargs)
        elif activation_type == "sine":
            return SineActivation(**kwargs)
        elif activation_type == "rational":
            return RationalActivation(**kwargs)
        elif activation_type == "wavelet":
            return WaveletActivation(**kwargs)
        elif activation_type == "rbf":
            return RBFActivation(**kwargs)
        else:
            raise ValueError(f"Tipo de activación no soportado: {activation_type}")
    
    def _create_fixed_activation(self, activation_name):
        """Crea función de activación fija"""
        if activation_name is None or activation_name == "linear":
            return lambda x: x
        elif activation_name == "relu":
            return torch.relu
        elif activation_name == "tanh":
            return torch.tanh
        elif activation_name == "sigmoid":
            return torch.sigmoid
        elif activation_name == "gelu":
            return torch.nn.functional.gelu
        elif activation_name == "swish" or activation_name == "silu":
            return torch.nn.functional.silu
        elif activation_name == "mish":
            return lambda x: x * torch.tanh(torch.nn.functional.softplus(x))
        elif activation_name == "leaky_relu":
            return torch.nn.functional.leaky_relu
        elif activation_name == "softmax":
            return lambda x: torch.nn.functional.softmax(x, dim=-1)
        elif activation_name == "abs":
            return torch.abs
        elif activation_name == "square":
            return lambda x: x ** 2
        elif activation_name == "sqrt":
            return lambda x: torch.sqrt(torch.abs(x) + 1e-8)
        elif activation_name == "exp":
            return torch.exp
        elif activation_name == "log":
            return lambda x: torch.log(torch.abs(x) + 1e-8)
        else:
            raise ValueError(f"Activación fija no soportada: {activation_name}")
    
    def _setup_input_functions(self, kwargs):
        """Configura las funciones de transformación de entrada"""
        
        if self.input_function_strategy == "none":
            self.input_activation_fn = None
            self.input_function_params = None
        
        elif self.input_function_strategy == "fixed":
            # Función de transformación fija para entradas
            self.input_activation_fn = self._create_fixed_activation(self.input_activation_type)
            self.input_function_params = None
            
        elif self.input_function_strategy == "global":
            # Una función compartida para todas las entradas
            self.input_activation_fn = self._create_activation_function(
                self.input_activation_type, **kwargs
            )
            self.input_function_params = nn.Parameter(
                torch.randn(1, self.input_activation_fn.n_params) * 0.1
            )
            
        elif self.input_function_strategy == "per_input":
            # Una función por característica de entrada
            self.input_activation_fn = self._create_activation_function(
                self.input_activation_type, **kwargs
            )
            self.input_function_params = nn.Parameter(
                torch.randn(self.in_features, self.input_activation_fn.n_params) * 0.1
            )
            
        elif self.input_function_strategy == "per_neuron_input":
            # Una función por cada entrada de cada neurona (máxima flexibilidad)
            self.input_activation_fn = self._create_activation_function(
                self.input_activation_type, **kwargs
            )
            self.input_function_params = nn.Parameter(
                torch.randn(self.in_features, self.out_features, 
                           self.input_activation_fn.n_params) * 0.1
            )
        else:
            raise ValueError(f"Estrategia de entrada no válida: {self.input_function_strategy}")
    
    def _setup_output_functions(self, kwargs):
        """Configura las funciones de activación de salida"""
        
        if self.output_function_strategy == "none":
            self.output_activation_fn = None
            self.output_function_params = None
            
        elif self.output_function_strategy == "fixed":
            # Función de activación fija - ahora usa output_activation_type
            self.output_activation_fn = self._create_fixed_activation(self.output_activation_type)
            self.output_function_params = None
            
        elif self.output_function_strategy == "global":
            # Una función compartida para todas las neuronas
            self.output_activation_fn = self._create_activation_function(
                self.output_activation_type, **kwargs
            )
            self.output_function_params = nn.Parameter(
                torch.randn(1, self.output_activation_fn.n_params) * 0.1
            )
            
        elif self.output_function_strategy == "per_neuron":
            # Una función por neurona
            self.output_activation_fn = self._create_activation_function(
                self.output_activation_type, **kwargs
            )
            self.output_function_params = nn.Parameter(
                torch.randn(self.out_features, self.output_activation_fn.n_params) * 0.1
            )
        else:
            raise ValueError(f"Estrategia de salida no válida: {self.output_function_strategy}")
    
    def _apply_regularization_before(self, x):
        """Aplica regularización antes de la activación"""
        if self.regularization_order == "dropout_first":
            # Dropout primero, luego batch norm
            if self.dropout_before is not None and self.training:
                x = self.dropout_before(x)
            if self.batch_norm_before is not None:
                x = self.batch_norm_before(x)
        else:  # batch_norm_first
            # Batch norm primero, luego dropout
            if self.batch_norm_before is not None:
                x = self.batch_norm_before(x)
            if self.dropout_before is not None and self.training:
                x = self.dropout_before(x)
        return x
    
    def _apply_regularization_after(self, x):
        """Aplica regularización después de la activación"""
        if self.regularization_order == "dropout_first":
            # Dropout primero, luego batch norm
            if self.dropout_after is not None and self.training:
                x = self.dropout_after(x)
            if self.batch_norm_after is not None:
                x = self.batch_norm_after(x)
        else:  # batch_norm_first
            # Batch norm primero, luego dropout
            if self.batch_norm_after is not None:
                x = self.batch_norm_after(x)
            if self.dropout_after is not None and self.training:
                x = self.dropout_after(x)
        return x
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Paso 1: Aplicar funciones de transformación de entrada
        if self.input_function_strategy == "none":
            transformed_inputs = x
        
        elif self.input_function_strategy == "fixed":
            # Aplicar función fija a todas las entradas
            transformed_inputs = self.input_activation_fn(x)
            
        elif self.input_function_strategy == "global":
            # Aplicar la misma función a todas las entradas
            transformed_inputs = torch.zeros_like(x)
            for i in range(self.in_features):
                transformed_inputs[:, i] = self.input_activation_fn(
                    x[:, i], self.input_function_params[0, :]
                )
                
        elif self.input_function_strategy == "per_input":
            # Aplicar función específica a cada entrada
            transformed_inputs = torch.zeros_like(x)
            for i in range(self.in_features):
                transformed_inputs[:, i] = self.input_activation_fn(
                    x[:, i], self.input_function_params[i, :]
                )
                
        elif self.input_function_strategy == "per_neuron_input":
            # Transformación específica para cada entrada de cada neurona
            # En este caso, calculamos directamente la contribución a cada neurona
            output = torch.zeros(batch_size, self.out_features, device=x.device)
            for i in range(self.in_features):
                for j in range(self.out_features):
                    transformed = self.input_activation_fn(
                        x[:, i], self.input_function_params[i, j, :]
                    )
                    output[:, j] += self.linear_weights[i, j] * transformed
            
            # Añadir bias
            linear_output = output + self.bias
            
            # Aplicar regularización antes de activación
            pre_activation = self._apply_regularization_before(linear_output)
            
            # Aplicar funciones de activación de salida
            activated_output = self._apply_output_activation(pre_activation)
            
            # Aplicar regularización después de activación
            final_output = self._apply_regularization_after(activated_output)
            
            return final_output
        
        # Paso 2: Combinación lineal (solo si no es per_neuron_input)
        linear_output = torch.matmul(transformed_inputs, self.linear_weights) + self.bias
        
        # Paso 3: Aplicar regularización antes de activación
        pre_activation = self._apply_regularization_before(linear_output)
        
        # Paso 4: Aplicar funciones de activación de salida
        activated_output = self._apply_output_activation(pre_activation)
        
        # Paso 5: Aplicar regularización después de activación
        final_output = self._apply_regularization_after(activated_output)
        
        return final_output
    
    def _apply_output_activation(self, linear_output):
        """Aplica las funciones de activación de salida"""
        
        if self.output_function_strategy == "none":
            return linear_output
            
        elif self.output_function_strategy == "fixed":
            return self.output_activation_fn(linear_output)
            
        elif self.output_function_strategy == "global":
            # Aplicar la misma función a todas las neuronas
            output = torch.zeros_like(linear_output)
            for j in range(self.out_features):
                output[:, j] = self.output_activation_fn(
                    linear_output[:, j], self.output_function_params[0, :]
                )
            return output
            
        elif self.output_function_strategy == "per_neuron":
            # Aplicar función específica a cada neurona
            output = torch.zeros_like(linear_output)
            for j in range(self.out_features):
                output[:, j] = self.output_activation_fn(
                    linear_output[:, j], self.output_function_params[j, :]
                )
            return output
    
    def count_parameters(self):
        """Cuenta parámetros de esta capa"""
        # Parámetros básicos
        linear_params = self.linear_weights.numel()
        bias_params = self.bias.numel()
        
        # Parámetros de funciones de entrada
        input_function_params = 0
        if self.input_function_params is not None:
            input_function_params = self.input_function_params.numel()
        
        # Parámetros de funciones de salida
        output_function_params = 0
        if self.output_function_params is not None:
            output_function_params = self.output_function_params.numel()
        
        # Parámetros de regularización
        batch_norm_params = 0
        if self.batch_norm_before is not None:
            batch_norm_params += sum(p.numel() for p in self.batch_norm_before.parameters())
        if self.batch_norm_after is not None:
            batch_norm_params += sum(p.numel() for p in self.batch_norm_after.parameters())
        
        total = linear_params + bias_params + input_function_params + output_function_params + batch_norm_params
        
        return {
            'linear_weights': linear_params,
            'bias': bias_params,
            'input_function_params': input_function_params,
            'output_function_params': output_function_params,
            'batch_norm_params': batch_norm_params,
            'total': total
        }
    
    def get_function_info(self):
        """Información sobre las funciones configuradas"""
        info = {
            'input_functions': [],
            'output_functions': []
        }
        
        # Información de funciones de entrada
        if self.input_function_strategy == "none":
            info['input_functions'].append("Sin transformación de entrada")
        elif self.input_function_strategy == "fixed":
            info['input_functions'].append(f"Fija: {self.input_activation_type}")
        elif self.input_function_strategy == "global":
            if hasattr(self.input_activation_fn, 'get_equation'):
                params = self.input_function_params[0, :]
                eq = f"φ_in_global(x) = {self.input_activation_fn.get_equation(params)}"
                info['input_functions'].append(eq)
            else:
                info['input_functions'].append(f"φ_in_global(x) = {self.input_activation_type}")
        elif self.input_function_strategy == "per_input":
            for i in range(self.in_features):
                if hasattr(self.input_activation_fn, 'get_equation'):
                    params = self.input_function_params[i, :]
                    eq = f"φ_in_{i}(x) = {self.input_activation_fn.get_equation(params)}"
                    info['input_functions'].append(eq)
                else:
                    info['input_functions'].append(f"φ_in_{i}(x) = {self.input_activation_type}")
        elif self.input_function_strategy == "per_neuron_input":
            for i in range(self.in_features):
                for j in range(self.out_features):
                    if hasattr(self.input_activation_fn, 'get_equation'):
                        params = self.input_function_params[i, j, :]
                        eq = f"φ_in_{i}→{j}(x) = {self.input_activation_fn.get_equation(params)}"
                        info['input_functions'].append(eq)
                    else:
                        info['input_functions'].append(f"φ_in_{i}→{j}(x) = {self.input_activation_type}")
        
        # Información de funciones de salida
        if self.output_function_strategy == "none":
            info['output_functions'].append("Sin activación de salida")
        elif self.output_function_strategy == "fixed":
            info['output_functions'].append(f"Fija: {self.output_activation_type}")
        elif self.output_function_strategy == "global":
            if hasattr(self.output_activation_fn, 'get_equation'):
                params = self.output_function_params[0, :]
                eq = f"φ_out_global(x) = {self.output_activation_fn.get_equation(params)}"
                info['output_functions'].append(eq)
            else:
                info['output_functions'].append(f"φ_out_global(x) = {self.output_activation_type}")
        elif self.output_function_strategy == "per_neuron":
            for j in range(self.out_features):
                if hasattr(self.output_activation_fn, 'get_equation'):
                    params = self.output_function_params[j, :]
                    eq = f"φ_out_{j}(x) = {self.output_activation_fn.get_equation(params)}"
                    info['output_functions'].append(eq)
                else:
                    info['output_functions'].append(f"φ_out_{j}(x) = {self.output_activation_type}")
        
        return info
    
    def get_regularization_info(self):
        """Información sobre la configuración de regularización"""
        info = {
            'dropout': {
                'probability': self.dropout_prob,
                'position': self.dropout_position,
                'before_activation': self.dropout_before is not None,
                'after_activation': self.dropout_after is not None
            },
            'batch_norm': {
                'enabled': self.use_batch_norm,
                'position': self.batch_norm_position,
                'before_activation': self.batch_norm_before is not None,
                'after_activation': self.batch_norm_after is not None
            },
            'order': self.regularization_order
        }
        return info


class DualFlexKAN(nn.Module):
    """
    Red KAN con control dual independiente por capa y regularización flexible
    """
    
    def __init__(self, layer_sizes, 
                 # Configuraciones por capa (listas)
                 input_function_strategies=None,
                 input_activation_types=None,
                 input_activation_kwargs=None,
                 output_function_strategies=None,
                 output_activation_types=None,
                 output_activation_kwargs=None,
                 # Regularización flexible por capa
                 dropout_probs=None,
                 dropout_positions=None,
                 use_batch_norms=None,
                 batch_norm_positions=None,
                 regularization_orders=None,
                 batch_norm_momentums=None,
                 batch_norm_epss=None
                 ):
        
        super(DualFlexKAN, self).__init__()
        
        n_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        
        # Valores por defecto
        if input_function_strategies is None:
            input_function_strategies = ["per_input"] * n_layers
        if input_activation_types is None:
            input_activation_types = ["polynomial"] * n_layers
        if input_activation_kwargs is None:
            input_activation_kwargs = [{}] * n_layers
        if output_function_strategies is None:
            output_function_strategies = ["per_neuron"] * n_layers
        if output_activation_types is None:
            output_activation_types = ["polynomial"] * n_layers
        if output_activation_kwargs is None:
            output_activation_kwargs = [{}] * n_layers
        if dropout_probs is None:
            dropout_probs = [0.0] * n_layers
        if dropout_positions is None:
            dropout_positions = ["after_activation"] * n_layers
        if use_batch_norms is None:
            use_batch_norms = [False] * n_layers
        if batch_norm_positions is None:
            batch_norm_positions = ["after_activation"] * n_layers
        if regularization_orders is None:
            regularization_orders = ["dropout_first"] * n_layers
        if batch_norm_momentums is None:
            batch_norm_momentums = [0.1] * n_layers
        if batch_norm_epss is None:
            batch_norm_epss = [1e-5] * n_layers
        
        # Almacenar configuraciones para métodos de información
        self.input_function_strategies = input_function_strategies
        self.input_activation_types = input_activation_types
        self.output_function_strategies = output_function_strategies
        self.output_activation_types = output_activation_types
        self.dropout_probs = dropout_probs
        self.dropout_positions = dropout_positions
        self.use_batch_norms = use_batch_norms
        self.batch_norm_positions = batch_norm_positions
        self.regularization_orders = regularization_orders
        
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            layer = DualFlexKANLinear(
                layer_sizes[i], 
                layer_sizes[i + 1],
                input_function_strategy=input_function_strategies[i],
                input_activation_type=input_activation_types[i],
                input_activation_kwargs=input_activation_kwargs[i],
                output_function_strategy=output_function_strategies[i],
                output_activation_type=output_activation_types[i],
                output_activation_kwargs=output_activation_kwargs[i],
                dropout_prob=dropout_probs[i],
                dropout_position=dropout_positions[i],
                use_batch_norm=use_batch_norms[i],
                batch_norm_position=batch_norm_positions[i],
                regularization_order=regularization_orders[i],
                batch_norm_momentum=batch_norm_momentums[i],
                batch_norm_eps=batch_norm_epss[i]
            )
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def count_parameters(self):
        """Cuenta parámetros totales de la red"""
        total_params = 0
        layer_params = []
        
        for i, layer in enumerate(self.layers):
            params = layer.count_parameters()
            layer_params.append(params)
            total_params += params['total']
        
        return {
            'total': total_params,
            'by_layer': layer_params
        }
    
    def print_architecture_info(self):
        """Muestra información detallada de la arquitectura"""
        
        print("\n=== Información de Arquitectura DualFlexKAN (Regularización Flexible) ===")
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            params = layer.count_parameters()
            reg_info = layer.get_regularization_info()
            total_params += params['total']
            
            # Determinar tipo de capa
            is_adaptive_input = layer.input_function_strategy not in ["none", "fixed"]
            is_adaptive_output = layer.output_function_strategy not in ["none", "fixed"]
            
            if is_adaptive_input and is_adaptive_output:
                layer_type = "KAN Dual (Adaptativa entrada + salida)"
            elif is_adaptive_input:
                layer_type = "KAN Híbrida (Adaptativa entrada)"
            elif is_adaptive_output:
                layer_type = "KAN Híbrida (Adaptativa salida)"
            else:
                layer_type = "MLP (Funciones fijas)"
            
            print(f"\nCapa {i+1} ({layer.in_features}→{layer.out_features}) - {layer_type}")
            
            # Información de entrada
            print(f"  Función entrada: {layer.input_activation_type} ({layer.input_function_strategy})")
            
            # Información de salida
            print(f"  Función salida: {layer.output_activation_type} ({layer.output_function_strategy})")
            
            # Información de regularización
            if reg_info['dropout']['probability'] > 0:
                print(f"  Dropout: {reg_info['dropout']['probability']} ({reg_info['dropout']['position']})")
            if reg_info['batch_norm']['enabled']:
                print(f"  BatchNorm: Activado ({reg_info['batch_norm']['position']})")
            if reg_info['dropout']['probability'] > 0 and reg_info['batch_norm']['enabled']:
                print(f"  Orden regularización: {reg_info['order']}")
            
            # Parámetros
            print(f"  Pesos lineales: {params['linear_weights']}")
            print(f"  Sesgos: {params['bias']}")
            if params['input_function_params'] > 0:
                print(f"  Parámetros función entrada: {params['input_function_params']}")
            if params['output_function_params'] > 0:
                print(f"  Parámetros función salida: {params['output_function_params']}")
            if params['batch_norm_params'] > 0:
                print(f"  Parámetros BatchNorm: {params['batch_norm_params']}")
            print(f"  Total capa: {params['total']} parámetros")
        
        print(f"\nTotal red: {total_params} parámetros")
        return total_params
    
    def print_learned_functions(self):
        """Imprime funciones aprendidas o fijas"""
        fdict = {}
        print("\n=== Funciones de Activación ===")
        
        for i, layer in enumerate(self.layers):
            print(f"\nCapa {i+1}:")
            
            # Funciones de entrada
            function_info = layer.get_function_info()
            
            print("  Funciones de entrada:")
            for j, func in enumerate(function_info['input_functions']):
                print(f"    {func}")
                fdict[(i+1, 'input', j+1)] = func
            
            print("  Funciones de salida:")
            for j, func in enumerate(function_info['output_functions']):
                print(f"    {func}")
                fdict[(i+1, 'output', j+1)] = func
        
        return fdict
    
    def print_regularization_summary(self):
        """Imprime resumen de la configuración de regularización"""
        print("\n=== Resumen de Regularización ===")
        
        for i, layer in enumerate(self.layers):
            reg_info = layer.get_regularization_info()
            print(f"\nCapa {i+1}:")
            
            # Información de dropout
            if reg_info['dropout']['probability'] > 0:
                print(f"  Dropout: {reg_info['dropout']['probability']}")
                print(f"    Posición: {reg_info['dropout']['position']}")
                print(f"    Antes activación: {reg_info['dropout']['before_activation']}")
                print(f"    Después activación: {reg_info['dropout']['after_activation']}")
            else:
                print("  Dropout: No aplicado")
            
            # Información de batch norm
            if reg_info['batch_norm']['enabled']:
                print(f"  BatchNorm: Activado")
                print(f"    Posición: {reg_info['batch_norm']['position']}")
                print(f"    Antes activación: {reg_info['batch_norm']['before_activation']}")
                print(f"    Después activación: {reg_info['batch_norm']['after_activation']}")
            else:
                print("  BatchNorm: No aplicado")
            
            # Orden de aplicación
            if reg_info['dropout']['probability'] > 0 and reg_info['batch_norm']['enabled']:
                print(f"  Orden de aplicación: {reg_info['order']}")


def polycoefs(polinomio_str):
    """Extrae coeficientes de un polinomio en formato string"""
    # Eliminamos el inicio (ej: 'φ_1(x) = ') y dividimos por '+'
    terminos = polinomio_str.split('=')[1].split('+')
    
    # Diccionario para guardar {grado: coeficiente}
    coeficientes = {}
    
    for term in terminos:
        term = term.strip()  # Eliminar espacios
        if 'x' in term:
            # Caso con x (ej: '0.334x' o '0.110x^3')
            partes = term.split('x')
            coef = float(partes[0])
            if '^' in partes[1]:
                grado = int(partes[1].split('^')[1])
            else:
                grado = 1
            coeficientes[grado] = coef
        else:
            # Término independiente (ej: '0.258')
            if term:  # Evitar strings vacíos
                coeficientes[0] = float(term)
    
    # Convertir a lista ordenada por grado
    if not coeficientes:
        return []
    max_grado = max(coeficientes.keys())
    return [coeficientes.get(grado, 0.0) for grado in range(max_grado + 1)]



#============================================================================
# Mecanismo de Atención en KAN
#============================================================================
class KANFeatureAttention(nn.Module):
    """
    Mecanismo de Atención de Características para KAN.
    Aprende a ponderar dinámicamente la importancia de cada característica de entrada
    antes de pasarla a la capa KAN principal.
    
    Arquitectura tipo "Squeeze-and-Excitation" modificada.
    """
    def __init__(self, in_features, reduction_ratio=4):
        super(KANFeatureAttention, self).__init__()
        
        reduced_dim = max(in_features // reduction_ratio, 1)
        
        # Red de contexto ligera para calcular los pesos de atención
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, in_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, in_features)
        
        # Calcular pesos de atención [0, 1] para cada característica
        attn_weights = self.attention_net(x)
        
        # Aplicar atención (escalado elemento a elemento)
        # Esto amplifica características relevantes y suprime ruido
        x_attended = x * attn_weights
        
        return x_attended, attn_weights

class DualFlexKANAttentionLayer(nn.Module):
    """
    Capa KAN con mecanismo de atención integrado.
    """
    def __init__(self, in_features, out_features, attention_reduction=4, **kan_kwargs):
        super(DualFlexKANAttentionLayer, self).__init__()
        
        # 1. Mecanismo de Atención
        self.attention = KANFeatureAttention(in_features, reduction_ratio=attention_reduction)
        
        # 2. Capa KAN estándar (DualFlexKANLinear)
        # Pasamos todos los argumentos de configuración (polinomios, estrategias, etc.)
        self.kan_layer = DualFlexKANLinear(in_features, out_features, **kan_kwargs)
        
    def forward(self, x):
        # 1. Aplicar atención a la entrada
        x_attended, weights = self.attention(x)
        
        # 2. Procesar la entrada ponderada con la capa KAN
        output = self.kan_layer(x_attended)
        
        return output, weights

class SelectiveAttentiveKAN(nn.Module):
    def __init__(self, layer_sizes, attention_layers_indices=None, attention_reduction=4, **kwargs):
        """
        Args:
            layer_sizes (list): Tamaños de las capas [input, hidden, ..., output]
            attention_layers_indices (list): Lista de índices de las capas que llevarán atención.
                                             Ej: [0] para solo la primera capa.
            attention_reduction (int): Factor de reducción para el módulo de atención.
            **kwargs: Argumentos para DualFlexKANLinear (estrategias, activaciones, etc.)
        """
        super(SelectiveAttentiveKAN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.attention_indices = set(attention_layers_indices) if attention_layers_indices else set()
        
        for i in range(len(layer_sizes) - 1):
            # Preparar kwargs específicos para esta capa si se pasaron listas
            layer_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, list) and len(value) == len(layer_sizes) - 1:
                    layer_kwargs[key] = value[i]
                else:
                    layer_kwargs[key] = value
            
            # DECISIÓN: ¿Esta capa lleva atención o es estándar?
            if i in self.attention_indices:
                # Instanciar capa con atención
                layer = DualFlexKANAttentionLayer(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i+1],
                    attention_reduction=attention_reduction,
                    **layer_kwargs
                )
            else:
                # Instanciar capa KAN estándar
                layer = DualFlexKANLinear(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i+1],
                    **layer_kwargs
                )
            
            self.layers.append(layer)
            
    def forward(self, x):
        self.attention_maps = {} # Diccionario para guardar pesos por capa
        
        for i, layer in enumerate(self.layers):
            if i in self.attention_indices:
                # La capa de atención devuelve (output, weights)
                x, weights = layer(x)
                self.attention_maps[i] = weights
            else:
                # La capa estándar devuelve solo output
                x = layer(x)
                
        return x

    
    '''
    Usar la capa con atención: en la definición de la capa, incluimos una DualFlexKANAttentionLayer
    
            layer = DualFlexKANAttentionLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i+1],
                attention_reduction=4, # Factor de compresión para la atención
                **kwargs # Pasa estrategias, tipos de activación, etc.
            )
    '''        
    
    
"""
if __name__ == "__main__":
    print("=== DualFlexKAN con estrategia 'fixed' en entrada ===")
    
    # Ejemplo con función fija de entrada (ReLU)
    model_fixed_input = DualFlexKAN(
        layer_sizes=[4, 16, 8, 2],
        input_function_strategies=["fixed", "fixed", "none"],
        input_activation_types=["relu", "abs", "polynomial"],
        output_function_strategies=["per_neuron", "fixed", "fixed"],
        output_activation_types=["rbf", "relu", "softmax"]
    )
    
    print("\nConfiguración con entrada 'fixed':")
    print("- Capa 1: ReLU fijo en entrada → RBF adaptativa en salida")
    print("- Capa 2: Abs fijo en entrada → ReLU fijo en salida")
    print("- Capa 3: Sin transformación entrada → Softmax fijo en salida")
    
    model_fixed_input.print_architecture_info()
    model_fixed_input.print_learned_functions()
    
    # Test forward pass
    x = torch.randn(32, 4)
    output = model_fixed_input(x)
    print(f"\nTest - Input: {x.shape} -> Output: {output.shape}")
    
    # Ejemplo comparando con entrada adaptativa
    print("\n" + "="*70)
    print("Comparación: Entrada fija vs adaptativa")
    print("="*70)
    
    model_adaptive_input = DualFlexKAN(
        layer_sizes=[4, 16, 8, 2],
        input_function_strategies=["per_input", "per_input", "none"],
        input_activation_types=["polynomial", "polynomial", "polynomial"],
        output_function_strategies=["per_neuron", "fixed", "fixed"],
        output_activation_types=["rbf", "relu", "softmax"]
    )
    
    params_fixed = model_fixed_input.count_parameters()
    params_adaptive = model_adaptive_input.count_parameters()
    
    print(f"\nParámetros totales:")
    print(f"- Modelo con entrada fija: {params_fixed['total']}")
    print(f"- Modelo con entrada adaptativa: {params_adaptive['total']}")
    print(f"- Diferencia: {params_adaptive['total'] - params_fixed['total']} parámetros")
    
    print("\nVentajas de usar 'fixed' en entrada:")
    print("✓ Menos parámetros (más eficiente)")
    print("✓ Transformaciones predefinidas y rápidas")
    print("✓ Útil para normalización o preprocesamiento")
    print("✓ Mayor estabilidad numérica")
    
    print("\nEjemplos de funciones fijas disponibles:")
    print("- relu, tanh, sigmoid, gelu, swish/silu, mish, leaky_relu")
    print("- abs, square, sqrt, exp, log")
    print("- linear (identidad), softmax")
"""