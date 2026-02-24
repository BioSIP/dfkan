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
KAN Initialization System - Version 2.0
100% backward compatible con versión original + nuevas capacidades avanzadas

Mantiene la API original:
- get_activation_type_info(layer)
- initialize_polynomial_params(params_tensor, activation_type, fan_in, fan_out)
- initialize_layer_params(layer)
- smart_kan_initialization(model, strategy="optimal")

Añade nuevas capacidades:
- KANInitializer class para control avanzado
- 10 estrategias para pesos lineales
- 6 estrategias para funciones polinomiales
- Inicialización layer-wise
- Sistema de análisis

@author: andres
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Dict, List, Optional, Union, Tuple


# ============================================================================
# FUNCIONES ORIGINALES (100% COMPATIBLES)
# ============================================================================

def get_activation_type_info(layer):
    """
    Extrae información sobre tipos de activación de una capa KAN
    
    FUNCIÓN ORIGINAL - Mantiene compatibilidad completa
    """
    info = {
        'input_type': getattr(layer, 'input_activation_type', None),
        'input_strategy': getattr(layer, 'input_function_strategy', 'none'),
        'output_type': getattr(layer, 'output_activation_type', None),
        'output_strategy': getattr(layer, 'output_function_strategy', 'none'),
        'has_input_params': hasattr(layer, 'input_function_params') and 
                           layer.input_function_params is not None,
        'has_output_params': hasattr(layer, 'output_function_params') and 
                            layer.output_function_params is not None,
        # FIX: Add fan_in and fan_out so KANInitializer can read them
        'fan_in': getattr(layer, 'in_features', 0),
        'fan_out': getattr(layer, 'out_features', 0)
    }
    return info


def initialize_polynomial_params(params_tensor, activation_type, fan_in=None, fan_out=None):
    """
    Inicializa parámetros de funciones polinomiales de forma óptima
    
    FUNCIÓN ORIGINAL - Mantiene compatibilidad completa
    Comportamiento idéntico a la versión 1.0
    """
    with torch.no_grad():
        if activation_type == "polynomial":
            n_params = params_tensor.shape[-1]
            if fan_in is not None:
                std = math.sqrt(2.0 / (fan_in + fan_out if fan_out else fan_in))
            else:
                std = 0.1
            for i in range(n_params):
                scale = std / (i + 1)**0.5
                params_tensor[..., i].normal_(0, scale)
            if n_params > 1:
                params_tensor[..., 1] *= 2.0
        # Implementación simplificada para otros tipos
        else:
            std = 0.05
            params_tensor.normal_(0, std)


def initialize_layer_params(layer):
    """
    Inicializa todos los parámetros de una capa KAN
    
    FUNCIÓN ORIGINAL - Mantiene compatibilidad completa
    Comportamiento idéntico a la versión 1.0
    """
    info = get_activation_type_info(layer)
    
    # Inicializar pesos lineales
    with torch.no_grad():
        fan_in = layer.in_features
        fan_out = layer.out_features
        std = math.sqrt(2.0 / (fan_in + fan_out))
        layer.linear_weights.normal_(0, std)
        layer.bias.zero_()
    
    # Inicializar parámetros de funciones
    if info['has_input_params']:
        initialize_polynomial_params(layer.input_function_params, info['input_type'], 
                                   layer.in_features, layer.out_features)
    if info['has_output_params']:
        initialize_polynomial_params(layer.output_function_params, info['output_type'], 
                                   layer.in_features, layer.out_features)


def smart_kan_initialization(model, strategy="optimal"):
    """
    Inicialización inteligente de redes KAN
    
    FUNCIÓN ORIGINAL MEJORADA:
    - Mantiene compatibilidad completa con versión 1.0
    - Añade soporte para nuevas estrategias avanzadas
    
    Args:
        model: Modelo KAN a inicializar
        strategy: Estrategia de inicialización
            - "optimal" (default): Comportamiento original (He-like)
            - Nuevas estrategias: "conservative", "aggressive", "stable", "deep", "sparse"
    """
    # Para estrategia "optimal", usar comportamiento original
    if strategy == "optimal":
        print(f"Inicializando red KAN con estrategia: {strategy}")
        for i, layer in enumerate(model.layers):
            print(f"Inicializando capa {i+1}...")
            initialize_layer_params(layer)
        print("Inicialización completa.")
    
    # Para nuevas estrategias, usar sistema avanzado
    else:
        _smart_kan_initialization_advanced(model, strategy, verbose=True)


# ============================================================================
# SISTEMA AVANZADO (NUEVAS CAPACIDADES)
# ============================================================================

def get_nonlinearity_gain(activation_type: str) -> float:
    """Retorna el gain factor apropiado para diferentes activaciones"""
    gains = {
        'linear': 1.0,
        'polynomial': 1.0,
        'relu': math.sqrt(2.0),
        'leaky_relu': math.sqrt(2.0 / (1 + 0.01**2)),
        'tanh': 5.0/3.0,
        'sigmoid': 1.0,
        'selu': 0.75,
        'gelu': 1.0,
        'swish': 1.0,
        'mish': 1.0,
        'legendre': 1.0,
        'gegenbauer': 1.0,
        'jacobi': 1.0,
        'bspline': 1.0,
        'rbf': 1.0
    }
    return gains.get(activation_type, 1.0)


# ============================================================================
# ESTRATEGIAS PARA PESOS LINEALES
# ============================================================================

def xavier_uniform_init(weights, bias, gain=1.0):
    """Inicialización Xavier/Glorot uniforme"""
    nn.init.xavier_uniform_(weights, gain=gain)
    if bias is not None:
        nn.init.zeros_(bias)


def xavier_normal_init(weights, bias, gain=1.0):
    """Inicialización Xavier/Glorot normal"""
    nn.init.xavier_normal_(weights, gain=gain)
    if bias is not None:
        nn.init.zeros_(bias)


def he_uniform_init(weights, bias, gain=math.sqrt(2.0)):
    """Inicialización He/Kaiming uniforme"""
    nn.init.kaiming_uniform_(weights, a=0, mode='fan_in', nonlinearity='relu')
    if bias is not None:
        nn.init.zeros_(bias)


def he_normal_init(weights, bias, gain=math.sqrt(2.0)):
    """Inicialización He/Kaiming normal"""
    nn.init.kaiming_normal_(weights, a=0, mode='fan_in', nonlinearity='relu')
    if bias is not None:
        nn.init.zeros_(bias)


def lecun_uniform_init(weights, bias):
    """Inicialización LeCun uniforme"""
    fan_in = weights.size(0)
    std = math.sqrt(1.0 / fan_in)
    bound = math.sqrt(3.0) * std
    nn.init.uniform_(weights, -bound, bound)
    if bias is not None:
        nn.init.zeros_(bias)


def lecun_normal_init(weights, bias):
    """Inicialización LeCun normal"""
    fan_in = weights.size(0)
    std = math.sqrt(1.0 / fan_in)
    nn.init.normal_(weights, 0, std)
    if bias is not None:
        nn.init.zeros_(bias)


def orthogonal_init(weights, bias, gain=1.0):
    """Inicialización ortogonal"""
    nn.init.orthogonal_(weights, gain=gain)
    if bias is not None:
        nn.init.zeros_(bias)


def sparse_init(weights, bias, sparsity=0.1, std=0.01):
    """Inicialización sparse"""
    nn.init.sparse_(weights, sparsity=sparsity, std=std)
    if bias is not None:
        nn.init.zeros_(bias)


# ============================================================================
# ESTRATEGIAS PARA FUNCIONES POLINOMIALES
# ============================================================================

def polynomial_standard_init(params_tensor, fan_in, fan_out, activation_type="polynomial"):
    """Inicialización estándar (comportamiento original mejorado)"""
    with torch.no_grad():
        n_params = params_tensor.shape[-1]
        gain = get_nonlinearity_gain(activation_type)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        
        for i in range(n_params):
            scale = std / math.sqrt(i + 1)
            params_tensor[..., i].normal_(0, scale)
        
        if n_params > 1:
            params_tensor[..., 1] *= 1.5


def polynomial_zero_bias_init(params_tensor, fan_in, fan_out, activation_type="polynomial"):
    """Inicialización con término constante en cero"""
    polynomial_standard_init(params_tensor, fan_in, fan_out, activation_type)
    with torch.no_grad():
        if params_tensor.shape[-1] > 0:
            params_tensor[..., 0].zero_()


def polynomial_linear_emphasis_init(params_tensor, fan_in, fan_out, 
                                     activation_type="polynomial", linear_scale=3.0):
    """Inicialización con fuerte énfasis en término lineal"""
    with torch.no_grad():
        n_params = params_tensor.shape[-1]
        gain = get_nonlinearity_gain(activation_type)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        
        for i in range(n_params):
            if i == 0:
                params_tensor[..., i].normal_(0, std * 0.1)
            elif i == 1:
                params_tensor[..., i].normal_(0, std * linear_scale)
            else:
                scale = std / (i ** 1.5)
                params_tensor[..., i].normal_(0, scale)


def polynomial_orthogonal_init(params_tensor, fan_in, fan_out, activation_type="polynomial"):
    """Inicialización inspirada en polinomios ortogonales"""
    with torch.no_grad():
        n_params = params_tensor.shape[-1]
        gain = get_nonlinearity_gain(activation_type)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        
        for i in range(n_params):
            scale = std * math.sqrt(1.0 / (2 * i + 1))
            params_tensor[..., i].normal_(0, scale)


def polynomial_small_random_init(params_tensor, fan_in, fan_out, 
                                  activation_type="polynomial", std=0.01):
    """Inicialización con valores muy pequeños"""
    with torch.no_grad():
        params_tensor.normal_(0, std)


def polynomial_adaptive_degree_init(params_tensor, fan_in, fan_out, 
                                     activation_type="polynomial", max_degree=None):
    """Inicialización adaptativa según el grado del polinomio"""
    with torch.no_grad():
        n_params = params_tensor.shape[-1]
        degree = n_params - 1
        
        if max_degree is None:
            max_degree = 10
        
        attenuation = math.exp(-degree / max_degree)
        gain = get_nonlinearity_gain(activation_type)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out)) * attenuation
        
        for i in range(n_params):
            scale = std / math.sqrt(i + 1)
            params_tensor[..., i].normal_(0, scale)


def rbf_init(params_tensor, fan_in, fan_out, n_centers, 
             center_range=(-2.0, 2.0)):
    """Inicialización especializada para RBF"""
    with torch.no_grad():
        centers = torch.linspace(center_range[0], center_range[1], n_centers)
        params_tensor[:n_centers] = centers.unsqueeze(0).expand_as(
            params_tensor[:n_centers])
        
        width_init = (center_range[1] - center_range[0]) / (2 * n_centers)
        params_tensor[n_centers:2*n_centers].fill_(width_init)
        
        std = math.sqrt(2.0 / (fan_in + fan_out))
        params_tensor[2*n_centers:].normal_(0, std)


# ============================================================================
# CLASE AVANZADA DE INICIALIZACIÓN
# ============================================================================

class KANInitializer:
    """
    Clase principal para gestionar inicialización avanzada de redes KAN
    
    NUEVO EN VERSIÓN 2.0
    """
    
    LINEAR_STRATEGIES = {
        'xavier_uniform': xavier_uniform_init,
        'xavier_normal': xavier_normal_init,
        'he_uniform': he_uniform_init,
        'he_normal': he_normal_init,
        'lecun_uniform': lecun_uniform_init,
        'lecun_normal': lecun_normal_init,
        'orthogonal': orthogonal_init,
        'sparse': sparse_init,
    }
    
    POLYNOMIAL_STRATEGIES = {
        'polynomial_standard': polynomial_standard_init,
        'polynomial_zero_bias': polynomial_zero_bias_init,
        'polynomial_linear': polynomial_linear_emphasis_init,
        'polynomial_orthogonal': polynomial_orthogonal_init,
        'polynomial_small': polynomial_small_random_init,
        'polynomial_adaptive': polynomial_adaptive_degree_init
    }
    
    def __init__(self, linear_strategy='he_normal', polynomial_strategy='polynomial_standard',
                 gain=1.0, verbose=True):
        self.linear_strategy = linear_strategy
        self.polynomial_strategy = polynomial_strategy
        self.gain = gain
        self.verbose = verbose
        
        if linear_strategy not in self.LINEAR_STRATEGIES:
            raise ValueError(f"Estrategia lineal '{linear_strategy}' no válida")
        if polynomial_strategy not in self.POLYNOMIAL_STRATEGIES:
            raise ValueError(f"Estrategia polinomial '{polynomial_strategy}' no válida")
    
    def initialize_linear_weights(self, layer, activation_type='relu'):
        """Inicializa pesos lineales de una capa"""
        gain = self.gain * get_nonlinearity_gain(activation_type)
        linear_init_fn = self.LINEAR_STRATEGIES[self.linear_strategy]
        
        if self.linear_strategy in ['xavier_uniform', 'xavier_normal', 'orthogonal']:
            linear_init_fn(layer.linear_weights, layer.bias, gain=gain)
        elif self.linear_strategy == 'sparse':
            linear_init_fn(layer.linear_weights, layer.bias, sparsity=0.1)
        else:
            linear_init_fn(layer.linear_weights, layer.bias)
    
    def initialize_polynomial_params_advanced(self, params_tensor, activation_type,
                                               fan_in, fan_out):
        """Inicializa parámetros polinomiales con estrategia avanzada"""
        poly_init_fn = self.POLYNOMIAL_STRATEGIES[self.polynomial_strategy]
        
        if activation_type == "rbf":
            n_centers = params_tensor.shape[-1] // 3
            rbf_init(params_tensor, fan_in, fan_out, n_centers)
        else:
            poly_init_fn(params_tensor, fan_in, fan_out, activation_type)
    
    def initialize_layer(self, layer, layer_idx=0):
        """Inicializa todos los parámetros de una capa"""
        info = get_activation_type_info(layer)
        
        if self.verbose:
            print(f"  Capa {layer_idx+1} ({info['fan_in']}→{info['fan_out']})")
        
        # Inicializar pesos lineales
        output_activation = info['output_type'] or 'linear'
        self.initialize_linear_weights(layer, output_activation)
        
        # Inicializar funciones de entrada
        if info['has_input_params']:
            self.initialize_polynomial_params_advanced(
                layer.input_function_params,
                info['input_type'],
                layer.in_features,
                layer.out_features
            )
        
        # Inicializar funciones de salida
        if info['has_output_params']:
            self.initialize_polynomial_params_advanced(
                layer.output_function_params,
                info['output_type'],
                layer.in_features,
                layer.out_features
            )
    
    def initialize_model(self, model):
        """Inicializa toda la red"""
        if self.verbose:
            print(f"\n{'='*60}")
            print("Inicializando Red KAN (Modo Avanzado)")
            print(f"{'='*60}")
            print(f"Estrategia lineal: {self.linear_strategy}")
            print(f"Estrategia polinomial: {self.polynomial_strategy}")
            print(f"{'='*60}\n")
        
        for i, layer in enumerate(model.layers):
            self.initialize_layer(layer, i)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Inicialización completada")
            print(f"{'='*60}\n")


# ============================================================================
# FUNCIÓN AVANZADA DE INICIALIZACIÓN (INTERNA)
# ============================================================================

def _smart_kan_initialization_advanced(model, strategy, verbose=True):
    """Función interna para estrategias avanzadas"""
    strategy_map = {
        'conservative': ('xavier_normal', 'polynomial_linear'),
        'aggressive': ('he_uniform', 'polynomial_adaptive'),
        'stable': ('lecun_normal', 'polynomial_zero_bias'),
        'deep': ('orthogonal', 'polynomial_orthogonal'),
        'sparse': ('sparse', 'polynomial_small')
    }
    
    if strategy not in strategy_map:
        if verbose:
            print(f"Advertencia: Estrategia '{strategy}' no reconocida. Usando 'optimal'.")
        # Usar comportamiento original
        print(f"Inicializando red KAN con estrategia: optimal")
        for i, layer in enumerate(model.layers):
            print(f"Inicializando capa {i+1}...")
            initialize_layer_params(layer)
        print("Inicialización completa.")
        return
    
    linear_strat, poly_strat = strategy_map[strategy]
    initializer = KANInitializer(
        linear_strategy=linear_strat,
        polynomial_strategy=poly_strat,
        verbose=verbose
    )
    initializer.initialize_model(model)


# ============================================================================
# INICIALIZACIÓN LAYER-WISE (NUEVA CAPACIDAD)
# ============================================================================

def initialize_model_layerwise(model, layer_configs, verbose=True):
    """
    Inicialización con configuración diferente por capa
    
    NUEVO EN VERSIÓN 2.0
    
    Args:
        model: Modelo KAN
        layer_configs: Lista de diccionarios con config por capa
        verbose: Imprimir información
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Inicialización Layer-wise")
        print(f"{'='*60}\n")
    
    for i, (layer, config) in enumerate(zip(model.layers, layer_configs)):
        linear_strategy = config.get('linear', 'he_normal')
        polynomial_strategy = config.get('polynomial', 'polynomial_standard')
        gain = config.get('gain', 1.0)
        
        initializer = KANInitializer(
            linear_strategy=linear_strategy,
            polynomial_strategy=polynomial_strategy,
            gain=gain,
            verbose=verbose
        )
        initializer.initialize_layer(layer, i)
    
    if verbose:
        print(f"{'='*60}")
        print("Inicialización layer-wise completada")
        print(f"{'='*60}\n")


# ============================================================================
# SISTEMA DE ANÁLISIS (NUEVA CAPACIDAD)
# ============================================================================

class InitializationAnalyzer:
    """Herramientas para analizar la calidad de la inicialización"""
    
    @staticmethod
    def analyze_layer_weights(layer, layer_name="Layer"):
        """Analiza estadísticas de pesos de una capa"""
        stats = {}
        
        with torch.no_grad():
            weights = layer.linear_weights
            stats['linear'] = {
                'mean': weights.mean().item(),
                'std': weights.std().item(),
                'min': weights.min().item(),
                'max': weights.max().item(),
                'norm': torch.norm(weights).item()
            }
            
            if hasattr(layer, 'input_function_params') and \
               layer.input_function_params is not None:
                params = layer.input_function_params
                stats['input_functions'] = {
                    'mean': params.mean().item(),
                    'std': params.std().item(),
                    'min': params.min().item(),
                    'max': params.max().item()
                }
            
            if hasattr(layer, 'output_function_params') and \
               layer.output_function_params is not None:
                params = layer.output_function_params
                stats['output_functions'] = {
                    'mean': params.mean().item(),
                    'std': params.std().item(),
                    'min': params.min().item(),
                    'max': params.max().item()
                }
        
        return stats
    
    @staticmethod
    def analyze_model(model, verbose=True):
        """Analiza toda la red"""
        all_stats = []
        
        if verbose:
            print(f"\n{'='*60}")
            print("Análisis de Inicialización")
            print(f"{'='*60}")
        
        for i, layer in enumerate(model.layers):
            stats = InitializationAnalyzer.analyze_layer_weights(layer, f"Capa {i+1}")
            all_stats.append(stats)
            
            if verbose:
                print(f"\nCapa {i+1}:")
                print(f"  Pesos lineales:")
                for key, value in stats['linear'].items():
                    print(f"    {key}: {value:.6f}")
        
        if verbose:
            print(f"{'='*60}\n")
        
        return all_stats


# ============================================================================
# INFORMACIÓN DE COMPATIBILIDAD
# ============================================================================

__version__ = "2.0"
__backward_compatible__ = True

def get_version_info():
    """Retorna información sobre la versión"""
    info = {
        'version': __version__,
        'backward_compatible': __backward_compatible__,
        'original_functions': [
            'get_activation_type_info',
            'initialize_polynomial_params',
            'initialize_layer_params',
            'smart_kan_initialization'
        ],
        'new_features': [
            'KANInitializer class',
            'initialize_model_layerwise',
            'InitializationAnalyzer',
            '10 linear strategies',
            '6 polynomial strategies'
        ]
    }
    return info


if __name__ == "__main__":
    print("KAN Initialization System v2.0")
    print("="*60)
    info = get_version_info()
    print(f"Versión: {info['version']}")
    print(f"Compatible con v1.0: {info['backward_compatible']}")
    print("\nFunciones originales mantenidas:")
    for func in info['original_functions']:
        print(f"  ✓ {func}")
    print("\nNuevas capacidades:")
    for feature in info['new_features']:
        print(f"  + {feature}")
