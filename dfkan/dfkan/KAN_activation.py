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
Created on Tue Jul  1 12:05:08 2025

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
from scipy.special import legendre, chebyt


# Funciones de activación estándar 
class StandardActivation(nn.Module):
    """
    Clase base para funciones de activación estándar
    """
    def __init__(self, activation_name="relu", **kwargs):
        super(StandardActivation, self).__init__()
        self.activation_name = activation_name
        self.n_params = 0  # Las funciones estándar no tienen parámetros aprendibles
        
        # Diccionario de funciones de activación
        self.activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=-1),
            "leaky_relu": nn.LeakyReLU(negative_slope=kwargs.get('negative_slope', 0.01)),
            "elu": nn.ELU(alpha=kwargs.get('alpha', 1.0)),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),  # SiLU es equivalente a Swish
            "mish": nn.Mish(),
            "linear": nn.Identity(),
        }
        
        if activation_name not in self.activations:
            raise ValueError(f"Función de activación '{activation_name}' no soportada. "
                           f"Opciones disponibles: {list(self.activations.keys())}")
        
        self.activation_fn = self.activations[activation_name]
    
    def forward(self, x, params=None):
        """
        Aplica la función de activación estándar
        params se ignora para funciones estándar
        """
        return self.activation_fn(x)
    
    def get_equation(self, params=None):
        """Retorna la ecuación de la función"""
        equations = {
            "relu": "max(0, x)",
            "sigmoid": "1/(1 + e^(-x))",
            "tanh": "tanh(x)",
            "softmax": "softmax(x)",
            "leaky_relu": "max(0.01*x, x)",
            "elu": "x if x>0 else α(e^x - 1)",
            "selu": "λ * (x if x>0 else α(e^x - 1))",
            "gelu": "x * Φ(x)",
            "swish": "x * sigmoid(x)",
            "mish": "x * tanh(softplus(x))",
            "linear": "x"
        }
        return equations.get(self.activation_name, f"{self.activation_name}(x)")



class ActivationFunction:
    """Clase base para funciones de activación"""
    def __init__(self, n_params):
        self.n_params = n_params
    
    def __call__(self, x, params):
        raise NotImplementedError
    
    def get_equation(self, params):
        raise NotImplementedError

class PolynomialActivation(ActivationFunction):
    """Polinomio cúbico estándar"""
    def __init__(self, degree=3):
        super().__init__(degree + 1)
        self.degree = degree
    
    def __call__(self, x, params):
        result = torch.zeros_like(x)
        for i in range(self.degree + 1):
            result += params[i] * (x ** i)
        return result
    
    def get_equation(self, params):
        terms = []
        for i in range(self.degree + 1):
            coeff = params[i].item()
            if i == 0:
                terms.append(f"{coeff:.3f}")
            elif i == 1:
                terms.append(f"{coeff:.3f}x")
            else:
                terms.append(f"{coeff:.3f}x^{i}")
        return " + ".join(terms)

class GegenbauerActivation(ActivationFunction):
    """Polinomios de Gegenbauer"""
    def __init__(self, max_degree=3, alpha=0.5):
        super().__init__(max_degree + 2)  # coeficientes + alpha + escala
        self.max_degree = max_degree
        self.alpha = alpha
    
    def __call__(self, x, params):
        # params[0] = alpha, params[1] = escala, params[2:] = coeficientes
        alpha = torch.abs(params[0]) + 0.1  # Evitar alpha <= 0
        scale = params[1]
        coeffs = params[2:]
        
        # Normalizar x para evitar overflow
        x_scaled = torch.tanh(x * scale)
        
        result = torch.zeros_like(x)
        for n in range(self.max_degree + 1):
            if n < len(coeffs):
                # Usar aproximación para evitar problemas numéricos
                gegenbauer_val = self._gegenbauer_approx(n, alpha, x_scaled)
                result += coeffs[n] * gegenbauer_val
        
        return result
    
    def _gegenbauer_approx(self, n, alpha, x):
        """Aproximación de polinomios de Gegenbauer usando recurrencia"""
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return 2 * alpha * x
        else:
            # Usar recurrencia: C_n^(α)(x) = (2(n+α-1)x*C_{n-1}^(α)(x) - (n+2α-2)*C_{n-2}^(α)(x)) / n
            c0 = torch.ones_like(x)
            c1 = 2 * alpha * x
            
            for i in range(2, n + 1):
                c_new = (2 * (i + alpha - 1) * x * c1 - (i + 2 * alpha - 2) * c0) / i
                c0, c1 = c1, c_new
            
            return c1
    
    def get_equation(self, params):
        alpha = abs(params[0].item()) + 0.1
        scale = params[1].item()
        return f"Gegenbauer(α={alpha:.3f}, scale={scale:.3f})"

class JacobiActivation(ActivationFunction):
    """Polinomios de Jacobi"""
    def __init__(self, max_degree=3, alpha=0.0, beta=0.0):
        super().__init__(max_degree + 3)  # coeficientes + alpha + beta + escala
        self.max_degree = max_degree
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, x, params):
        alpha = params[0]
        beta = params[1] 
        scale = params[2]
        coeffs = params[3:]
        
        # Normalizar x al rango [-1, 1]
        x_scaled = torch.tanh(x * scale)
        
        result = torch.zeros_like(x)
        for n in range(self.max_degree + 1):
            if n < len(coeffs):
                jacobi_val = self._jacobi_approx(n, alpha, beta, x_scaled)
                result += coeffs[n] * jacobi_val
        
        return result
    
    def _jacobi_approx(self, n, alpha, beta, x):
        """Aproximación de polinomios de Jacobi usando recurrencia"""
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return (alpha - beta + (alpha + beta + 2) * x) / 2
        else:
            p0 = torch.ones_like(x)
            p1 = (alpha - beta + (alpha + beta + 2) * x) / 2
            
            for i in range(2, n + 1):
                a1 = 2 * i * (i + alpha + beta) * (2 * i + alpha + beta - 2)
                a2 = (2 * i + alpha + beta - 1) * (alpha**2 - beta**2)
                a3 = (2 * i + alpha + beta - 1) * (2 * i + alpha + beta) * (2 * i + alpha + beta - 2)
                a4 = 2 * (i + alpha - 1) * (i + beta - 1) * (2 * i + alpha + beta)
                
                p_new = ((a2 + a3 * x) * p1 - a4 * p0) / a1
                p0, p1 = p1, p_new
            
            return p1
    
    def get_equation(self, params):
        alpha = params[0].item()
        beta = params[1].item()
        scale = params[2].item()
        return f"Jacobi(α={alpha:.3f}, β={beta:.3f}, scale={scale:.3f})"

class BSplineActivation(ActivationFunction):
    """
    B-splines con orden configurable usando la fórmula de Cox-de Boor
    
    Args:
        n_knots: Número de knots internos (puntos de control)
        order: Orden del B-spline (1=constante, 2=lineal, 3=cuadrático, 4=cúbico)
               Note: degree = order - 1
    """
    def __init__(self, n_knots=5, order=3):
        # Número de parámetros: coeficientes para cada función base + scale + offset
        # Para B-splines: necesitamos n_knots + order coeficientes
        super().__init__(n_knots + order + 2)  # +2 para scale y offset
        self.n_knots = n_knots
        self.order = order
        self.degree = order - 1
        
    def _create_knot_vector(self, x_min, x_max):
        """
        Crea un vector de knots extendido para B-splines
        Usa knots uniformes con repetición en los extremos
        """
        # Knots internos equiespaciados
        internal_knots = torch.linspace(x_min, x_max, self.n_knots)
        
        # Repetir extremos según el orden (condición de frontera estándar)
        left_padding = torch.full((self.order,), x_min, device=internal_knots.device)
        right_padding = torch.full((self.order,), x_max, device=internal_knots.device)
        
        # Vector de knots completo: [t_0, ..., t_0, t_1, ..., t_n, t_n, ..., t_n]
        #                             order veces    internos     order veces
        knot_vector = torch.cat([left_padding, internal_knots, right_padding])
        
        return knot_vector
    
    def _cox_de_boor(self, x, i, k, knots):
        """
        Calcula la función base B-spline B_{i,k}(x) usando recurrencia de Cox-de Boor
        
        B_{i,1}(x) = 1 si t_i <= x < t_{i+1}, 0 en otro caso
        B_{i,k}(x) = (x - t_i)/(t_{i+k-1} - t_i) * B_{i,k-1}(x) + 
                     (t_{i+k} - x)/(t_{i+k} - t_{i+1}) * B_{i+1,k-1}(x)
        
        Args:
            x: Puntos donde evaluar
            i: Índice de la función base
            k: Orden de la función base
            knots: Vector de knots
        """
        # Caso base: B-spline de orden 1 (constante a trozos)
        if k == 1:
            # Evitar problemas en los bordes
            mask = (x >= knots[i]) & (x < knots[i + 1])
            # Incluir el último punto
            if i == len(knots) - k - 1:
                mask = mask | (x == knots[i + 1])
            return mask.float()
        
        # Caso recursivo
        # Primer término: (x - t_i) / (t_{i+k-1} - t_i) * B_{i,k-1}(x)
        denom1 = knots[i + k - 1] - knots[i]
        if torch.abs(denom1) < 1e-10:
            term1 = torch.zeros_like(x)
        else:
            term1 = ((x - knots[i]) / denom1) * self._cox_de_boor(x, i, k - 1, knots)
        
        # Segundo término: (t_{i+k} - x) / (t_{i+k} - t_{i+1}) * B_{i+1,k-1}(x)
        denom2 = knots[i + k] - knots[i + 1]
        if torch.abs(denom2) < 1e-10:
            term2 = torch.zeros_like(x)
        else:
            term2 = ((knots[i + k] - x) / denom2) * self._cox_de_boor(x, i + 1, k - 1, knots)
        
        return term1 + term2
    
    def __call__(self, x, params):
        """
        Evalúa la combinación lineal de funciones base B-spline
        
        Args:
            x: Input tensor
            params: [scale, offset, coef_0, coef_1, ..., coef_{n+k-1}]
        """
        scale = params[0]
        offset = params[1]
        coeffs = params[2:]
        
        # Escalar y desplazar entrada
        x_scaled = x * scale + offset
        
        # Crear vector de knots (rango fijo para estabilidad)
        x_min, x_max = -3.0, 3.0
        knots = self._create_knot_vector(x_min, x_max)
        
        # Número de funciones base
        n_basis = self.n_knots + self.order
        
        # Evaluar combinación lineal de funciones base
        result = torch.zeros_like(x)
        
        for i in range(min(n_basis, len(coeffs))):
            # Evaluar i-ésima función base B-spline
            basis_i = self._cox_de_boor(x_scaled, i, self.order, knots)
            result += coeffs[i] * basis_i
        
        return result
    
    def get_equation(self, params):
        scale = params[0].item()
        offset = params[1].item()
        return f"B-Spline(order={self.order}, degree={self.degree}, n_knots={self.n_knots}, scale={scale:.3f})"


class FastBSplineActivation(ActivationFunction):
    """
    Versión optimizada de B-splines usando aproximación por funciones gaussianas
    Más rápida pero menos precisa que Cox-de Boor
    Útil para entrenamiento rápido
    """
    def __init__(self, n_knots=5, order=3):
        super().__init__(n_knots + order + 2)
        self.n_knots = n_knots
        self.order = order
        self.degree = order - 1
    
    def __call__(self, x, params):
        scale = params[0]
        offset = params[1]
        coeffs = params[2:]
        
        x_scaled = x * scale + offset
        
        # Knots equiespaciados
        knots = torch.linspace(-3, 3, self.n_knots + self.order, device=x.device)
        
        result = torch.zeros_like(x)
        
        # Ancho de la función base depende del orden
        width = 2.0 / (self.order + 1)
        
        for i, coeff in enumerate(coeffs[:len(knots)]):
            # Distancia al knot
            dist = torch.abs(x_scaled - knots[i])
            
            # Función base según orden (aproximación)
            if self.order == 1:  # Constante
                basis = (dist < width).float()
            
            elif self.order == 2:  # Lineal (hat function)
                basis = torch.clamp(1.0 - dist / width, 0.0, 1.0)
            
            elif self.order == 3:  # Cuadrático
                basis = torch.where(
                    dist < 1.5 * width,
                    torch.clamp(1.0 - (dist / (1.5 * width)) ** 2, 0.0, 1.0),
                    torch.zeros_like(dist)
                )
            
            elif self.order == 4:  # Cúbico
                basis = torch.where(
                    dist < 2.0 * width,
                    torch.clamp(1.0 - (dist / (2.0 * width)) ** 3, 0.0, 1.0),
                    torch.zeros_like(dist)
                )
            
            else:  # order >= 5
                # Aproximación gaussiana suavizada
                basis = torch.exp(-0.5 * (dist / width) ** 2)
            
            result += coeff * basis
        
        return result
    
    def get_equation(self, params):
        scale = params[0].item()
        offset = params[1].item()
        return f"FastB-Spline(order={self.order}, n_knots={self.n_knots}, scale={scale:.3f})"

class RBFActivation(ActivationFunction):
    """Funciones de Base Radial (Gaussianas)"""
    def __init__(self, n_centers=4):
        super().__init__(n_centers * 3)  # centers, widths, weights
        self.n_centers = n_centers
    
    def __call__(self, x, params):
        # Reorganizar parámetros: [centers, widths, weights]
        centers = params[:self.n_centers]
        widths = torch.abs(params[self.n_centers:2*self.n_centers]) + 0.1
        weights = params[2*self.n_centers:]
        
        result = torch.zeros_like(x)
        for i in range(self.n_centers):
            # Función gaussiana
            dist_sq = (x - centers[i]) ** 2
            rbf_val = torch.exp(-dist_sq / (2 * widths[i] ** 2))
            result += weights[i] * rbf_val
        
        return result
    
    def get_equation(self, params):
        return f"RBF({self.n_centers} gaussianas)"
    
    
### Poliniomios de Legendre normalizados para usar una base ortonormal ###
class LegendreActivation(nn.Module):
    """
    Función de activación basada en polinomios de Legendre normalizados
    f(x) = Σ(i=0 to k) c_i * P_i(x)
    donde P_i son los polinomios de Legendre normalizados
    """
    def __init__(self, order=3, input_range=(-1, 1)):
        super(LegendreActivation, self).__init__()
        self.order = order
        self.n_params = order + 1  # c_0, c_1, ..., c_k
        self.input_range = input_range
        
        # Precomputar coeficientes de los polinomios de Legendre hasta orden k
        self.legendre_coeffs = self._precompute_legendre_coeffs()
        
    def _precompute_legendre_coeffs(self):
        """Precomputa los coeficientes de los polinomios de Legendre"""
        coeffs = []
        for i in range(self.order + 1):
            # Usar scipy para obtener coeficientes del polinomio de Legendre P_i
            poly = legendre(i)
            coeffs.append(torch.tensor(poly.coeffs[::-1], dtype=torch.float32))  # Invertir para orden ascendente
        return coeffs
    
    def _normalize_input(self, x):
        """Normaliza la entrada al rango [-1, 1] para los polinomios de Legendre"""
        a, b = self.input_range
        return 2 * (x - a) / (b - a) - 1
    
    def _evaluate_legendre(self, x, order):
        """Evalúa el polinomio de Legendre P_order(x) usando los coeficientes precomputados"""
        if order >= len(self.legendre_coeffs):
            return torch.zeros_like(x)
        
        coeffs = self.legendre_coeffs[order].to(x.device)
        result = torch.zeros_like(x)
        
        # Evaluar polinomio usando esquema de Horner
        for i, coeff in enumerate(coeffs):
            result += coeff * (x ** i)
        
        return result
    
    def forward(self, x, params):
        """
        Evalúa la combinación lineal de polinomios de Legendre
        Args:
            x: tensor de entrada
            params: coeficientes [c_0, c_1, ..., c_k]
        """
        # Normalizar entrada al rango [-1, 1]
        x_norm = self._normalize_input(x)
        
        # Evaluar suma ponderada de polinomios de Legendre
        result = torch.zeros_like(x)
        for i in range(self.order + 1):
            if i < len(params):
                legendre_val = self._evaluate_legendre(x_norm, i)
                result += params[i] * legendre_val
        
        return result
    
    def get_equation(self, params=None):
        """Genera la ecuación de la función"""
        if params is None:
            return f"Σ(i=0 to {self.order}) c_i * P_i(x)"
        
        terms = []
        for i in range(min(len(params), self.order + 1)):
            coeff = params[i].item() if hasattr(params[i], 'item') else params[i]
            if abs(coeff) > 1e-6:  # Solo mostrar términos significativos
                terms.append(f"{coeff:.3f}*P_{i}(x)")
        
        return " + ".join(terms) if terms else "0"

class OptimizedLegendreActivation(nn.Module):
    """
    Versión optimizada usando recurrencia para calcular polinomios de Legendre
    P_0(x) = 1
    P_1(x) = x
    P_{n+1}(x) = ((2n+1)x*P_n(x) - n*P_{n-1}(x)) / (n+1)
    """
    def __init__(self, order=3, input_range=(-1, 1)):
        super(OptimizedLegendreActivation, self).__init__()
        self.order = order
        self.n_params = order + 1
        self.input_range = input_range
        
    def _normalize_input(self, x):
        """Normaliza la entrada al rango [-1, 1]"""
        a, b = self.input_range
        return 2 * (x - a) / (b - a) - 1
    
    def _evaluate_legendre_recursive(self, x_norm, order):
        """Evalúa P_order(x) usando recurrencia"""
        if order == 0:
            return torch.ones_like(x_norm)
        elif order == 1:
            return x_norm
        
        # Inicializar P_0 y P_1
        P_prev2 = torch.ones_like(x_norm)  # P_0
        P_prev1 = x_norm                   # P_1
        
        # Calcular P_2, P_3, ..., P_order usando recurrencia
        for n in range(1, order):
            P_current = ((2*n + 1) * x_norm * P_prev1 - n * P_prev2) / (n + 1)
            P_prev2, P_prev1 = P_prev1, P_current
        
        return P_prev1
    
    def forward(self, x, params):
        """Evalúa la combinación lineal de polinomios de Legendre"""
        x_norm = self._normalize_input(x)
        
        result = torch.zeros_like(x)
        for i in range(self.order + 1):
            if i < len(params):
                legendre_val = self._evaluate_legendre_recursive(x_norm, i)
                result += params[i] * legendre_val
        
        return result
    
    def get_equation(self, params=None):
        """Genera la ecuación de la función"""
        if params is None:
            return f"Σ(i=0 to {self.order}) c_i * P_i(x)"
        
        terms = []
        for i in range(min(len(params), self.order + 1)):
            coeff = params[i].item() if hasattr(params[i], 'item') else params[i]
            if abs(coeff) > 1e-6:
                terms.append(f"{coeff:.3f}*P_{i}(x)")
        
        return " + ".join(terms) if terms else "0"


class ChebyshevActivation(nn.Module):
    """
    Polinomios de Chebyshev de primera especie T_n(x).
    Definidos por recurrencia: T_0=1, T_1=x, T_{n+1} = 2xT_n - T_{n-1}
    Eficiente y ortogonal en [-1, 1].
    """
    def __init__(self, degree=3, input_range=(-1, 1)):
        super(ChebyshevActivation, self).__init__()
        self.degree = degree
        self.n_params = degree + 1
        self.input_range = input_range

    def _normalize_input(self, x):
        """Normaliza x a [-1, 1]"""
        a, b = self.input_range
        return 2.0 * (x - a) / (b - a) - 1.0

    def forward(self, x, params):
        x_norm = self._normalize_input(x)
        
        # Caso base vectorizado
        T_prev2 = torch.ones_like(x_norm) # T_0
        T_prev1 = x_norm                  # T_1
        
        result = torch.zeros_like(x_norm)
        
        # c_0 * T_0
        if len(params) > 0:
            result += params[0] * T_prev2
        
        # c_1 * T_1
        if len(params) > 1:
            result += params[1] * T_prev1
            
        # Recurrencia eficiente para n >= 2
        for n in range(2, self.degree + 1):
            # T_n = 2*x*T_{n-1} - T_{n-2}
            T_curr = 2 * x_norm * T_prev1 - T_prev2
            
            if n < len(params):
                result += params[n] * T_curr
            
            # Actualizar para siguiente iteración
            T_prev2, T_prev1 = T_prev1, T_curr
            
        return result

    def get_equation(self, params=None):
        if params is None:
            return f"Σ c_i * T_i(x) (Chebyshev deg={self.degree})"
        terms = []
        for i in range(min(len(params), self.degree + 1)):
            coeff = params[i].item() if hasattr(params[i], 'item') else params[i]
            if abs(coeff) > 1e-4:
                terms.append(f"{coeff:.3f}T_{i}")
        return " + ".join(terms) if terms else "0"
    

class SineActivation(ActivationFunction):
    """
    Base de Fourier (SineKAN).
    f(x) = Σ [a_k * sin(k*x) + b_k * cos(k*x)]
    Implementación vectorizada para alta eficiencia.
    """
    def __init__(self, num_freqs=5):
        # 2 parámetros por frecuencia (sin y cos) + término constante opcional
        super().__init__(2 * num_freqs + 1)
        self.num_freqs = num_freqs
        
        # Buffer de frecuencias [1, 2, ..., K] para broadcasting
        # Se registra como buffer para que se mueva a GPU automáticamente con el modelo
        self.register_buffer('freqs', torch.arange(1, num_freqs + 1, dtype=torch.float32))

    def __call__(self, x, params):
        # params layout: [bias, a_1, b_1, a_2, b_2, ...]
        bias = params[0]
        coeffs = params[1:]
        
        # Separar coeficientes a (seno) y b (coseno)
        # Asumiendo params tiene tamaño 2*num_freqs
        n_coeffs = len(coeffs)
        # Asegurar que sea par para reshape
        if n_coeffs % 2 != 0:
            n_coeffs -= 1
            
        # Reshape para operaciones vectorizadas
        # coeffs_matrix shape: [num_freqs, 2] -> col 0: sin, col 1: cos
        coeffs_pairs = coeffs[:n_coeffs].view(-1, 2)
        
        # Operación vectorizada:
        # 1. Expandir x: [Batch, 1]
        x_exp = x.unsqueeze(-1) 
        
        # 2. Calcular argumentos: k*x para todas las k a la vez
        # self.freqs: [num_freqs]
        # angles: [Batch, num_freqs]
        angles = x_exp * self.freqs[:len(coeffs_pairs)]
        
        # 3. Calcular bases
        sines = torch.sin(angles)
        cosines = torch.cos(angles)
        
        # 4. Suma ponderada
        # Sum (a_k * sin + b_k * cos)
        result = bias + torch.sum(
            coeffs_pairs[:, 0] * sines + coeffs_pairs[:, 1] * cosines, 
            dim=-1
        )
        
        return result

    def get_equation(self, params):
        return f"Fourier(K={self.num_freqs}) + bias"
    
    # Necesario para register_buffer en __init__
    def register_buffer(self, name, tensor):
        if not hasattr(self, '_buffers'):
            self._buffers = {}
        self._buffers[name] = tensor
    
    # Sobrescribimos __getattr__ para acceder al buffer simulando nn.Module si se usa como ActivationFunction pura
    def __getattr__(self, name):
        if '_buffers' in self.__dict__ and name in self._buffers:
            return self._buffers[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        

class RationalActivation(ActivationFunction):
    """
    Funciones Racionales (Pade Approximants).
    f(x) = P(x) / Q(x)
    Donde P y Q son polinomios.
    Se asegura Q(x) != 0 usando valor absoluto y epsilon.
    """
    def __init__(self, degree_numerator=3, degree_denominator=2):
        super().__init__(degree_numerator + 1 + degree_denominator + 1)
        self.deg_p = degree_numerator
        self.deg_q = degree_denominator
    
    def __call__(self, x, params):
        # Dividir parámetros entre P y Q
        p_coeffs = params[:self.deg_p + 1]
        q_coeffs = params[self.deg_p + 1:]
        
        # Evaluar P(x) usando esquema de Horner (estable y rápido)
        num = torch.zeros_like(x)
        # P(x) = a_0 + a_1*x + ... + a_n*x^n
        # Horner: a_0 + x*(a_1 + x*(...))
        for i in range(len(p_coeffs) - 1, -1, -1):
            num = num * x + p_coeffs[i]
            
        # Evaluar Q(x)
        # Nota: Forzamos el término constante de Q a ser 1 (o algo positivo) 
        # y coeficientes positivos o epsilon para evitar singularidades
        denom = torch.zeros_like(x)
        
        # Q(x) = |b_0| + |b_1|x + ...
        # Se suele fijar b_0 = 1 para evitar redundancia de escala, 
        # pero aquí dejamos que sea aprendible pero positivo.
        
        epsilon = 1e-4 # Evitar división por cero
        
        for i in range(len(q_coeffs) - 1, -1, -1):
            # Usamos abs para estabilidad
            coeff = torch.abs(q_coeffs[i])
            if i == 0: 
                coeff = coeff + 1.0 # Forzar que el término independiente sea >= 1
            denom = denom * x + coeff
            
        # Asegurar denominador seguro
        denom = torch.where(torch.abs(denom) < epsilon, torch.ones_like(denom) * epsilon, denom)
        
        return num / denom

    def get_equation(self, params):
        return f"Rational P({self.deg_p}) / Q({self.deg_q})"
    
class WaveletActivation(ActivationFunction):
    """
    Base de Wavelets (Mexican Hat / Ricker).
    psi(t) = (1 - t^2) * exp(-t^2 / 2)
    f(x) = Σ w_i * psi((x - mu_i) / s_i)
    
    Implementación eficiente precalculando una rejilla de mu y s.
    """
    def __init__(self, n_wavelets=5):
        # 3 params por wavelet si aprendemos mu/sigma/weight, 
        # pero para KAN estable, solemos fijar mu/sigma (grid) y aprender pesos.
        # Aquí implementamos la versión flexible: pesos aprendibles sobre grid fijo,
        # más parámetros de escala global.
        super().__init__(n_wavelets + 2) # Pesos + scale_global + translation_global
        self.n_wavelets = n_wavelets
        
        # Grid fijo para estabilidad inicial (como en B-Splines)
        # Centros distribuidos en [-3, 3]
        self.register_buffer('centers', torch.linspace(-3.0, 3.0, n_wavelets))
        # Anchuras (sigmas) fijas iniciales
        self.register_buffer('sigmas', torch.ones(n_wavelets) * 1.0)

    def __call__(self, x, params):
        # params: [scale_in, offset_in, w_1, w_2, ..., w_n]
        scale_in = params[0]
        offset_in = params[1]
        weights = params[2:]
        
        # Transformar entrada
        x_prime = x * scale_in + offset_in
        
        # Mexican Hat Wavelet: (1 - t^2) * exp(-0.5 * t^2)
        # Vectorización: [Batch, 1] - [1, n_wavelets] -> [Batch, n_wavelets]
        
        # t = (x - mu) / sigma
        # x_prime: [B] -> [B, 1]
        t = (x_prime.unsqueeze(-1) - self.centers) / self.sigmas
        
        # Evaluar base wavelet (fórmula Ricker)
        # psi = (1 - t^2) * exp(-0.5 * t^2)
        psi = (1 - t**2) * torch.exp(-0.5 * t**2)
        
        # Combinación lineal: [B, n_wavelets] * [n_wavelets] -> [B, n_wavelets] -> sum -> [B]
        # Aseguramos dimension correcta de weights
        n_w = min(len(weights), self.n_wavelets)
        w_active = weights[:n_w]
        
        result = torch.sum(psi[:, :n_w] * w_active, dim=-1)
        
        return result

    def get_equation(self, params):
        return f"Wavelet Sum (Mexican Hat, N={self.n_wavelets})"

    def register_buffer(self, name, tensor):
        if not hasattr(self, '_buffers'):
            self._buffers = {}
        self._buffers[name] = tensor
    
    def __getattr__(self, name):
        if '_buffers' in self.__dict__ and name in self._buffers:
            return self._buffers[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        
"""
# Ejemplo de uso con polinomios de Legendre
def example_legendre_kan():
    
    print("=== Ejemplo KAN con Polinomios de Legendre ===")
    
    # Crear red con diferentes tipos de activación
    layer_sizes = [3, 8, 4, 1]
    activation_types = ["legendre", "polynomial", "legendre"]
    sharing_strategies = ["per_input", "per_output", "global"]
    activation_kwargs = [
        {"order": 4, "input_range": (-2, 2)},  # Legendre orden 4
        {"degree": 3},                         # Polinomio cúbico
        {"order": 3, "input_range": (-1, 1)}   # Legendre orden 3
    ]
    
    model = FlexKAN(
        layer_sizes=layer_sizes,
        activation_types=activation_types,
        sharing_strategies=sharing_strategies,
        activation_kwargs=activation_kwargs,
        is_fixed_activations=[False, False, False]
    )
    
    # Información de la arquitectura
    total_params = model.print_architecture_info()
    
    # Funciones aprendidas
    functions_dict = model.print_learned_functions()
    
    # Prueba con datos de ejemplo
    x = torch.randn(32, 3)
    y = model(x)
    print(f"\nEntrada: {x.shape}")
    print(f"Salida: {y.shape}")
    
    return model, functions_dict
"""