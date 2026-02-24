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
Análisis de Interacciones para Redes KAN

Herramientas para analizar cómo las funciones aprendidas interactúan entre sí:
1. Mapas de interacción entre funciones y neuronas
2. Descomposición funcional (contribución individual de cada φᵢ)
3. Análisis de pares de interacciones
4. Comparación de estrategias (global vs per_input vs per_neuron)
5. Análisis de sinergia entre features

@author: andres
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


class KANInteractionAnalysis:
    """
    Sistema de análisis de interacciones para redes KAN
    
    Analiza cómo las funciones univariadas φᵢ(xᵢ) se combinan e interactúan
    para producir las salidas de la red.
    """
    
    def __init__(self, model, feature_names=None):
        """
        Args:
            model: Modelo DualFlexKAN entrenado
            feature_names: Lista con nombres de features (opcional)
        """
        self.model = model
        self.n_layers = len(model.layers)
        
        if feature_names is None:
            n_features = model.layers[0].in_features
            self.feature_names = [f"x_{i}" for i in range(n_features)]
        else:
            self.feature_names = feature_names
    
    # ========================================================================
    # 1. DESCOMPOSICIÓN FUNCIONAL
    # ========================================================================
    
    def functional_decomposition(self, X_sample, layer_idx=0, neuron_idx=None):
        """
        Descompone la salida en contribuciones individuales: y = Σᵢ wᵢ·φᵢ(xᵢ)
        
        Args:
            X_sample: Datos de entrada (N, n_features)
            layer_idx: Índice de capa a analizar
            neuron_idx: Neurona específica (None = todas)
        
        Returns:
            dict: {
                'contributions': array (N, in_features, out_features),
                'function_outputs': transformaciones φᵢ(xᵢ),
                'weighted_contributions': wᵢ·φᵢ(xᵢ),
                'total_output': suma total por neurona
            }
        """
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.FloatTensor(X_sample)
        
        layer = self.model.layers[layer_idx]
        n_samples = X_sample.shape[0]
        
        # Arrays para almacenar contribuciones
        function_outputs = torch.zeros(n_samples, layer.in_features)
        contributions = torch.zeros(n_samples, layer.in_features, layer.out_features)
        
        with torch.no_grad():
            # 1. Aplicar funciones de entrada φᵢ(xᵢ)
            if hasattr(layer, 'input_function_params') and \
               layer.input_function_params is not None:
                
                for i in range(layer.in_features):
                    x_i = X_sample[:, i:i+1]
                    
                    if layer.input_function_strategy == "per_input":
                        params = layer.input_function_params[i]
                    elif layer.input_function_strategy == "global":
                        params = layer.input_function_params[0]
                    else:
                        # Sin transformación
                        function_outputs[:, i] = x_i.squeeze()
                        continue
                    
                    # Aplicar función
                    phi_i = layer.input_activation_fn(x_i, params)
                    function_outputs[:, i] = phi_i.squeeze()
            else:
                # Sin funciones, usar inputs directamente
                function_outputs = X_sample
            
            # 2. Multiplicar por pesos: wᵢ·φᵢ(xᵢ)
            weights = layer.linear_weights  # (in_features, out_features)
            
            for i in range(layer.in_features):
                for j in range(layer.out_features):
                    contributions[:, i, j] = function_outputs[:, i] * weights[i, j]
            
            # 3. Sumar para obtener salida total (sin bias ni activación de salida)
            total_output = contributions.sum(dim=1)  # (N, out_features)
        
        # Seleccionar neurona específica si se solicita
        if neuron_idx is not None:
            contributions = contributions[:, :, neuron_idx:neuron_idx+1]
            total_output = total_output[:, neuron_idx:neuron_idx+1]
        
        return {
            'contributions': contributions.numpy(),
            'function_outputs': function_outputs.numpy(),
            'weighted_contributions': contributions.numpy(),
            'total_output': total_output.numpy(),
            'weights': layer.linear_weights.detach().numpy(),
            'layer_idx': layer_idx,
            'neuron_idx': neuron_idx
        }
    
    def plot_functional_decomposition(self, decomp_results, sample_idx=0, 
                                     neuron_idx=0, save_path=None):
        """
        Visualiza la descomposición funcional para una muestra específica
        """
        contributions = decomp_results['contributions']
        total = decomp_results['total_output']
        layer_idx = decomp_results['layer_idx']
        
        # Contribuciones de cada feature para la muestra seleccionada
        sample_contrib = contributions[sample_idx, :, neuron_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Barras de contribución absoluta
        ax = axes[0, 0]
        x_pos = np.arange(len(sample_contrib))
        colors = ['green' if c > 0 else 'red' for c in sample_contrib]
        bars = ax.bar(x_pos, sample_contrib, color=colors, alpha=0.7, edgecolor='black')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Feature', fontsize=11, fontweight='bold')
        ax.set_ylabel('Contribución wᵢ·φᵢ(xᵢ)', fontsize=11, fontweight='bold')
        ax.set_title(f'Contribuciones Individuales - Neurona {neuron_idx}',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.feature_names[i] if i < len(self.feature_names)
                           else f'x_{i}' for i in range(len(sample_contrib))],
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores
        for i, (bar, val) in enumerate(zip(bars, sample_contrib)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top',
                   fontsize=8)
        
        # 2. Contribución relativa (%)
        ax = axes[0, 1]
        abs_contrib = np.abs(sample_contrib)
        rel_contrib = abs_contrib / (abs_contrib.sum() + 1e-8) * 100
        
        bars = ax.barh(x_pos, rel_contrib, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_yticks(x_pos)
        ax.set_yticklabels([self.feature_names[i] if i < len(self.feature_names)
                           else f'x_{i}' for i in range(len(sample_contrib))])
        ax.set_xlabel('Contribución Relativa (%)', fontsize=11, fontweight='bold')
        ax.set_title('Importancia Relativa por Feature', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Añadir valores
        for bar, val in zip(bars, rel_contrib):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {val:.1f}%', ha='left', va='center', fontsize=9)
        
        # 3. Waterfall chart - acumulación de contribuciones
        ax = axes[1, 0]
        
        # Ordenar por magnitud de contribución
        sorted_indices = np.argsort(np.abs(sample_contrib))[::-1]
        sorted_contrib = sample_contrib[sorted_indices]
        sorted_names = [self.feature_names[i] if i < len(self.feature_names)
                       else f'x_{i}' for i in sorted_indices]
        
        cumulative = np.cumsum(sorted_contrib)
        
        # Inicio en 0
        x_pos_water = np.arange(len(sorted_contrib) + 1)
        y_water = np.concatenate([[0], cumulative])
        
        # Dibujar barras
        for i in range(len(sorted_contrib)):
            color = 'green' if sorted_contrib[i] > 0 else 'red'
            ax.bar(i+1, sorted_contrib[i], bottom=y_water[i],
                  color=color, alpha=0.7, edgecolor='black')
        
        # Línea acumulativa
        ax.plot(x_pos_water, y_water, 'ko-', linewidth=2, markersize=6)
        
        # Línea final (total)
        ax.axhline(y=y_water[-1], color='blue', linestyle='--',
                  linewidth=2, label=f'Total = {y_water[-1]:.2f}')
        
        ax.set_xlabel('Feature (ordenado por importancia)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Contribución Acumulada', fontsize=11, fontweight='bold')
        ax.set_title('Waterfall Chart - Acumulación de Contribuciones',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos_water)
        ax.set_xticklabels(['0'] + sorted_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Heatmap de funciones φᵢ(xᵢ) vs pesos wᵢ
        ax = axes[1, 1]
        
        func_outputs = decomp_results['function_outputs'][sample_idx]
        weights = decomp_results['weights'][:, neuron_idx]
        
        # Crear matriz para visualizar
        comparison = np.column_stack([func_outputs, weights, sample_contrib])
        
        im = ax.imshow(comparison.T, aspect='auto', cmap='RdBu_r', 
                      interpolation='nearest')
        
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['φᵢ(xᵢ)', 'wᵢ', 'wᵢ·φᵢ(xᵢ)'], fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.feature_names[i] if i < len(self.feature_names)
                           else f'x_{i}' for i in range(len(sample_contrib))],
                          rotation=45, ha='right')
        ax.set_title('Función vs Peso vs Contribución', fontsize=12, fontweight='bold')
        
        # Añadir valores
        for i in range(comparison.shape[0]):
            for j in range(3):
                text = ax.text(i, j, f'{comparison[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Valor')
        
        plt.suptitle(f'Descomposición Funcional - Capa {layer_idx}, ' +
                    f'Muestra {sample_idx}, Neurona {neuron_idx}',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # 2. MAPAS DE INTERACCIÓN
    # ========================================================================
    
    def compute_interaction_map(self, X_sample, layer_idx=0):
        """
        Calcula mapa de interacción: cómo cada input afecta cada neurona
        
        Returns:
            dict: {
                'interaction_strength': matriz (in_features, out_features),
                'interaction_type': positivo/negativo,
                'normalized_map': normalizado por neurona
            }
        """
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.FloatTensor(X_sample)
        
        # Obtener descomposición funcional
        decomp = self.functional_decomposition(X_sample, layer_idx)
        contributions = decomp['contributions']  # (N, in_features, out_features)
        
        # Promediar sobre las muestras
        avg_contrib = np.mean(contributions, axis=0)  # (in_features, out_features)
        
        # Calcular "fuerza" de interacción (magnitud promedio)
        interaction_strength = np.abs(avg_contrib)
        
        # Tipo de interacción (positiva o negativa en promedio)
        interaction_type = np.sign(avg_contrib)
        
        # Normalizar por neurona (suma = 1 para cada neurona)
        normalized_map = interaction_strength / (interaction_strength.sum(axis=0, keepdims=True) + 1e-8)
        
        return {
            'interaction_strength': interaction_strength,
            'interaction_type': interaction_type,
            'avg_contribution': avg_contrib,
            'normalized_map': normalized_map,
            'layer_idx': layer_idx
        }
    
    def plot_interaction_map(self, interaction_results, save_path=None):
        """Visualiza mapas de interacción"""
        
        strength = interaction_results['interaction_strength']
        interaction_type = interaction_results['interaction_type']
        avg_contrib = interaction_results['avg_contribution']
        normalized = interaction_results['normalized_map']
        layer_idx = interaction_results['layer_idx']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Mapa de fuerza de interacción
        ax = axes[0, 0]
        im = ax.imshow(strength, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Neurona de Salida', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature de Entrada', fontsize=11, fontweight='bold')
        ax.set_title('Fuerza de Interacción (magnitud)', fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_yticklabels(self.feature_names)
        plt.colorbar(im, ax=ax, label='|Contribución|')
        
        # 2. Tipo de interacción (positiva/negativa)
        ax = axes[0, 1]
        im = ax.imshow(interaction_type, aspect='auto', cmap='RdBu_r',
                      vmin=-1, vmax=1, interpolation='nearest')
        ax.set_xlabel('Neurona de Salida', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature de Entrada', fontsize=11, fontweight='bold')
        ax.set_title('Tipo de Interacción', fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_yticklabels(self.feature_names)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['Negativa', 'Neutral', 'Positiva'])
        
        # 3. Contribución promedio con signo
        ax = axes[1, 0]
        im = ax.imshow(avg_contrib, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax.set_xlabel('Neurona de Salida', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature de Entrada', fontsize=11, fontweight='bold')
        ax.set_title('Contribución Promedio (con signo)', fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_yticklabels(self.feature_names)
        plt.colorbar(im, ax=ax, label='wᵢ·φᵢ(xᵢ)')
        
        # Añadir valores si matriz es pequeña
        if avg_contrib.shape[0] <= 10 and avg_contrib.shape[1] <= 10:
            for i in range(avg_contrib.shape[0]):
                for j in range(avg_contrib.shape[1]):
                    text = ax.text(j, i, f'{avg_contrib[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        # 4. Importancia normalizada por neurona
        ax = axes[1, 1]
        im = ax.imshow(normalized, aspect='auto', cmap='Greens', interpolation='nearest')
        ax.set_xlabel('Neurona de Salida', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature de Entrada', fontsize=11, fontweight='bold')
        ax.set_title('Importancia Relativa (normalizada)', fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_yticklabels(self.feature_names)
        plt.colorbar(im, ax=ax, label='Proporción')
        
        plt.suptitle(f'Mapas de Interacción - Capa {layer_idx}',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # 3. ANÁLISIS DE PARES DE INTERACCIONES
    # ========================================================================
    
    def pairwise_interaction_analysis(self, X_sample, layer_idx=0,
                                     feature_pairs=None, top_k=5):
        """
        Analiza interacciones entre pares de features
        
        Examina si dos features trabajan juntas (sinergia) o en oposición
        
        Args:
            X_sample: Datos de entrada
            layer_idx: Capa a analizar
            feature_pairs: Lista de tuplas (i, j) o None (auto-detectar top k)
            top_k: Si feature_pairs=None, número de pares más fuertes
        
        Returns:
            dict con análisis de pares
        """
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.FloatTensor(X_sample)
        
        decomp = self.functional_decomposition(X_sample, layer_idx)
        contributions = decomp['contributions']  # (N, in_features, out_features)
        
        n_features = contributions.shape[1]
        n_neurons = contributions.shape[2]
        
        # Auto-detectar pares si no se especifican
        if feature_pairs is None:
            # Calcular "sinergia" para cada par (correlación de contribuciones)
            pair_scores = []
            for i, j in combinations(range(n_features), 2):
                # Para cada neurona, ver si i y j contribuyen de forma similar
                contrib_i = contributions[:, i, :]  # (N, out_features)
                contrib_j = contributions[:, j, :]  # (N, out_features)
                
                # Correlación promedio sobre neuronas
                corr_per_neuron = []
                for n in range(n_neurons):
                    if np.std(contrib_i[:, n]) > 1e-8 and np.std(contrib_j[:, n]) > 1e-8:
                        corr = np.corrcoef(contrib_i[:, n], contrib_j[:, n])[0, 1]
                        corr_per_neuron.append(abs(corr))
                
                avg_corr = np.mean(corr_per_neuron) if corr_per_neuron else 0
                pair_scores.append(((i, j), avg_corr))
            
            # Seleccionar top k pares
            pair_scores.sort(key=lambda x: x[1], reverse=True)
            feature_pairs = [p[0] for p in pair_scores[:top_k]]
        
        # Analizar cada par
        pair_results = []
        
        for feat_i, feat_j in feature_pairs:
            pair_info = {
                'features': (feat_i, feat_j),
                'names': (self.feature_names[feat_i], self.feature_names[feat_j]),
                'correlations': [],
                'joint_contribution': [],
                'synergy_score': 0
            }
            
            for neuron_idx in range(n_neurons):
                contrib_i = contributions[:, feat_i, neuron_idx]
                contrib_j = contributions[:, feat_j, neuron_idx]
                
                # Correlación
                if np.std(contrib_i) > 1e-8 and np.std(contrib_j) > 1e-8:
                    corr = np.corrcoef(contrib_i, contrib_j)[0, 1]
                else:
                    corr = 0
                
                pair_info['correlations'].append(corr)
                
                # Contribución conjunta
                joint = contrib_i + contrib_j
                pair_info['joint_contribution'].append(np.mean(np.abs(joint)))
            
            # Score de sinergia (correlación promedio)
            pair_info['synergy_score'] = np.mean([abs(c) for c in pair_info['correlations']])
            pair_info['avg_correlation'] = np.mean(pair_info['correlations'])
            
            pair_results.append(pair_info)
        
        return {
            'pairs': pair_results,
            'layer_idx': layer_idx,
            'n_features': n_features,
            'n_neurons': n_neurons
        }
    
    def plot_pairwise_interactions(self, pairwise_results, save_path=None):
        """Visualiza análisis de pares"""
        
        pairs = pairwise_results['pairs']
        layer_idx = pairwise_results['layer_idx']
        n_pairs = len(pairs)
        
        if n_pairs == 0:
            print("No hay pares para visualizar")
            return
        
        fig, axes = plt.subplots(2, min(3, n_pairs), figsize=(5*min(3, n_pairs), 10))
        if n_pairs == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, pair_info in enumerate(pairs[:6]):  # Máximo 6 pares
            row = idx // 3
            col = idx % 3
            
            if idx >= 6:
                break
            
            ax = axes[row, col] if n_pairs > 1 else axes[row]
            
            feat_i, feat_j = pair_info['features']
            name_i, name_j = pair_info['names']
            correlations = pair_info['correlations']
            
            # Gráfica de correlaciones por neurona
            x_pos = np.arange(len(correlations))
            colors = ['green' if c > 0 else 'red' for c in correlations]
            
            ax.bar(x_pos, correlations, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Neurona', fontsize=10)
            ax.set_ylabel('Correlación', fontsize=10)
            ax.set_title(f'{name_i} ↔ {name_j}\n' +
                        f'Sinergia: {pair_info["synergy_score"]:.3f}',
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([-1.1, 1.1])
        
        plt.suptitle(f'Interacciones por Pares - Capa {layer_idx}',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # 4. COMPARACIÓN DE ESTRATEGIAS
    # ========================================================================
    
    def compare_strategies(self, X_sample, layer_idx=0):
        """
        Compara qué tan similares/diferentes son las funciones aprendidas
        según la estrategia (global vs per_input)
        
        Returns:
            dict con métricas de diversidad funcional
        """
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.FloatTensor(X_sample)
        
        layer = self.model.layers[layer_idx]
        strategy = layer.input_function_strategy
        
        if not hasattr(layer, 'input_function_params') or \
           layer.input_function_params is None:
            print(f"Capa {layer_idx} no tiene funciones aprendibles")
            return None
        
        params = layer.input_function_params.detach()
        
        results = {
            'strategy': strategy,
            'layer_idx': layer_idx,
            'diversity_metrics': {}
        }
        
        if strategy == 'global':
            # Solo una función compartida
            results['n_functions'] = 1
            results['diversity_metrics']['variance'] = 0
            results['diversity_metrics']['diversity_score'] = 0
            results['message'] = "Estrategia global: todas las features usan la misma función"
        
        elif strategy == 'per_input':
            # Una función por feature
            n_functions = params.shape[0]
            results['n_functions'] = n_functions
            
            # Calcular similitud entre pares de funciones
            similarities = []
            x_test = torch.linspace(-3, 3, 100)
            
            function_outputs = []
            for i in range(n_functions):
                y_i = layer.input_activation_fn(x_test.unsqueeze(1), params[i])
                function_outputs.append(y_i.squeeze().numpy())
            
            # Correlación entre funciones
            for i in range(n_functions):
                for j in range(i+1, n_functions):
                    if np.std(function_outputs[i]) > 1e-8 and np.std(function_outputs[j]) > 1e-8:
                        corr = np.corrcoef(function_outputs[i], function_outputs[j])[0, 1]
                        similarities.append(abs(corr))
            
            avg_similarity = np.mean(similarities) if similarities else 0
            results['diversity_metrics']['avg_similarity'] = avg_similarity
            results['diversity_metrics']['diversity_score'] = 1 - avg_similarity
            
            # Varianza de parámetros
            param_variance = params.var(dim=0).mean().item()
            results['diversity_metrics']['param_variance'] = param_variance
            
            results['message'] = f"Estrategia per_input: {n_functions} funciones con " + \
                               f"diversidad {results['diversity_metrics']['diversity_score']:.3f}"
        
        return results
    
    def plot_strategy_comparison(self, strategy_results, save_path=None):
        """Visualiza comparación de estrategias"""
        
        if strategy_results is None:
            return
        
        strategy = strategy_results['strategy']
        layer_idx = strategy_results['layer_idx']
        
        if strategy == 'global':
            print(f"Capa {layer_idx} usa estrategia global - solo una función")
            return
        
        diversity = strategy_results['diversity_metrics']['diversity_score']
        avg_sim = strategy_results['diversity_metrics']['avg_similarity']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Gráfica de barras de métricas
        metrics = ['Diversidad', 'Similitud Promedio']
        values = [diversity, avg_sim]
        colors = ['green', 'orange']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Análisis de Estrategia - Capa {layer_idx} ({strategy})',
                    fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Interpretación
        interp_text = f"Interpretación:\n"
        if diversity > 0.7:
            interp_text += "• Alta diversidad: funciones muy especializadas\n"
        elif diversity > 0.4:
            interp_text += "• Diversidad moderada: algunas funciones similares\n"
        else:
            interp_text += "• Baja diversidad: funciones parecidas (¿estrategia global mejor?)\n"
        
        ax.text(0.5, 0.5, interp_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # 5. ANÁLISIS COMPLETO DE INTERACCIONES
    # ========================================================================
    
    def comprehensive_interaction_analysis(self, X_sample, layer_idx=0,
                                          save_dir='./interaction_analysis'):
        """
        Ejecuta todos los análisis de interacciones y genera reporte completo
        
        Args:
            X_sample: Muestra de datos
            layer_idx: Capa a analizar
            save_dir: Directorio para guardar resultados
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("="*70)
        print("ANÁLISIS COMPLETO DE INTERACCIONES - KAN")
        print("="*70)
        
        # 1. Descomposición funcional
        print(f"\n1. Descomposición funcional...")
        decomp = self.functional_decomposition(X_sample, layer_idx)
        self.plot_functional_decomposition(
            decomp,
            sample_idx=0,
            neuron_idx=0,
            save_path=f'{save_dir}/functional_decomposition.png'
        )
        print(f"   ✓ Guardado en {save_dir}/functional_decomposition.png")
        
        # 2. Mapas de interacción
        print(f"\n2. Mapas de interacción...")
        interaction_map = self.compute_interaction_map(X_sample, layer_idx)
        self.plot_interaction_map(
            interaction_map,
            save_path=f'{save_dir}/interaction_map.png'
        )
        print(f"   ✓ Guardado en {save_dir}/interaction_map.png")
        
        # 3. Análisis de pares
        print(f"\n3. Análisis de pares de interacciones...")
        pairwise = self.pairwise_interaction_analysis(X_sample, layer_idx, top_k=6)
        self.plot_pairwise_interactions(
            pairwise,
            save_path=f'{save_dir}/pairwise_interactions.png'
        )
        print(f"   ✓ Guardado en {save_dir}/pairwise_interactions.png")
        
        # 4. Comparación de estrategias
        print(f"\n4. Comparación de estrategias...")
        strategy_comp = self.compare_strategies(X_sample, layer_idx)
        if strategy_comp:
            self.plot_strategy_comparison(
                strategy_comp,
                save_path=f'{save_dir}/strategy_comparison.png'
            )
            print(f"   ✓ Guardado en {save_dir}/strategy_comparison.png")
        
        # Resumen
        print("\n" + "="*70)
        print("RESUMEN DE INTERACCIONES")
        print("="*70)
        
        # Top contribuciones por feature
        print(f"\nTop 3 Features por contribución promedio:")
        avg_contrib = interaction_map['avg_contribution']
        feature_totals = np.abs(avg_contrib).sum(axis=1)
        top_features = np.argsort(feature_totals)[::-1][:3]
        
        for i, idx in enumerate(top_features, 1):
            fname = self.feature_names[idx] if idx < len(self.feature_names) else f'x_{idx}'
            print(f"  {i}. {fname}: {feature_totals[idx]:.4f}")
        
        # Top pares sinérgicos
        if pairwise['pairs']:
            print(f"\nTop 3 Pares más sinérgicos:")
            sorted_pairs = sorted(pairwise['pairs'], 
                                key=lambda x: x['synergy_score'], reverse=True)
            for i, pair in enumerate(sorted_pairs[:3], 1):
                print(f"  {i}. {pair['names'][0]} ↔ {pair['names'][1]}: {pair['synergy_score']:.4f}")
        
        # Diversidad funcional
        if strategy_comp and 'diversity_score' in strategy_comp['diversity_metrics']:
            div_score = strategy_comp['diversity_metrics']['diversity_score']
            print(f"\nDiversidad funcional: {div_score:.4f}")
            if div_score > 0.7:
                print("  → Funciones altamente especializadas")
            elif div_score > 0.4:
                print("  → Funciones moderadamente diferenciadas")
            else:
                print("  → Funciones similares (baja especialización)")
        
        print("\n" + "="*70)
        print(f"Análisis completo guardado en: {save_dir}")
        print("="*70)
        
        return {
            'decomposition': decomp,
            'interaction_map': interaction_map,
            'pairwise': pairwise,
            'strategy_comparison': strategy_comp
        }



