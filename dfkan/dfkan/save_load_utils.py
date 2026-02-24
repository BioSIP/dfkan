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
save_load_utils.py
==================
Utilidades para guardar y cargar objetos Python de forma universal.
"""

import pickle
import json
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import importlib
import inspect
import types
import sys
import textwrap
import types
import dill


def save_object(obj, filepath, format='pickle', overwrite=False, verbose=True):
    """
    Guarda un objeto Python en disco
    
    Parámetros:
        obj: Objeto a guardar
        filepath (str): Ruta del archivo
        format (str): 'pickle' o 'json'
        overwrite (bool): Si True, sobrescribe
        verbose (bool): Si True, imprime info
    
    Retorna:
        bool: True si éxito
    
    Ejemplo:
        save_object(metrics, 'results/metrics.pkl')
    """
    if os.path.exists(filepath) and not overwrite:
        print(f"El archivo '{filepath}' ya existe. Usa overwrite=True para sobrescribir.")
        return False
    
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        if verbose:
            print(f"Directorio creado: {directory}")
    
    try:
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif format == 'json':
            obj_serializable = _make_json_serializable(obj)
            with open(filepath, 'w') as f:
                json.dump(obj_serializable, f, indent=4)
        
        else:
            raise ValueError(f"Formato no soportado: {format}. Usa 'pickle' o 'json'")
        
        if verbose:
            file_size = os.path.getsize(filepath) / 1024
            print(f"Objeto guardado: {filepath} ({file_size:.2f} KB)")
        
        return True
    
    except Exception as e:
        print(f"Error al guardar: {e}")
        return False


def load_object(filepath, format='auto', verbose=True):
    """
    Carga un objeto desde disco
    
    Parámetros:
        filepath (str): Ruta del archivo
        format (str): 'pickle', 'json' o 'auto'
        verbose (bool): Si True, imprime info
    
    Retorna:
        object: El objeto cargado o None si error
    
    Ejemplo:
        metrics = load_object('results/metrics.pkl')
    """
    if not os.path.exists(filepath):
        print(f"El archivo '{filepath}' no existe.")
        return None
    
    if format == 'auto':
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.pkl', '.pickle']:
            format = 'pickle'
        elif ext == '.json':
            format = 'json'
        else:
            print(f" Extensión '{ext}' no reconocida. Intentando pickle...")
            format = 'pickle'
    
    try:
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
        
        elif format == 'json':
            with open(filepath, 'r') as f:
                obj = json.load(f)
        
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        if verbose:
            file_size = os.path.getsize(filepath) / 1024
            print(f"Objeto cargado: {filepath} ({file_size:.2f} KB)")
        
        return obj
    
    except Exception as e:
        print(f"Error al cargar: {e}")
        return None


def save_multiple(filepath, overwrite=False, verbose=True, **objects):
    """
    Guarda múltiples objetos en un solo archivo pickle
    
    Parámetros:
        filepath (str): Ruta del archivo .pkl
        overwrite (bool): Si True, sobrescribe
        verbose (bool): Si True, imprime info
        **objects: Objetos a guardar como kwargs
    
    Retorna:
        bool: True si éxito
    
    Ejemplo:
        save_multiple('results/exp.pkl', metrics={'acc': 0.95}, config={'lr': 0.001})
    """
    if not objects:
        print(" No se proporcionaron objetos para guardar.")
        return False
    
    if not filepath.endswith('.pkl') and not filepath.endswith('.pickle'):
        filepath += '.pkl'
    
    return save_object(objects, filepath, format='pickle', 
                      overwrite=overwrite, verbose=verbose)


def load_multiple(filepath, verbose=True):
    """
    Carga múltiples objetos desde un archivo pickle
    
    Parámetros:
        filepath (str): Ruta del archivo .pkl
        verbose (bool): Si True, imprime info
    
    Retorna:
        dict: Diccionario con los objetos cargados
    
    Ejemplo:
        data = load_multiple('results/exp.pkl')
        metrics = data['metrics']
    """
    obj = load_object(filepath, format='pickle', verbose=verbose)
    
    if obj and verbose and isinstance(obj, dict):
        print(f"Objetos cargados: {list(obj.keys())}")
    
    return obj


def _make_json_serializable(obj):
    """Convierte objetos numpy a tipos serializables por JSON"""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_json_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_experiment(experiment_name, save_dir='experiments', 
                   metrics=None, config=None, history=None, 
                   model_state=None, **kwargs):
    """
    Guarda un experimento completo con timestamp
    
    Parámetros:
        experiment_name (str): Nombre del experimento
        save_dir (str): Directorio base
        metrics (dict): Métricas finales
        config (dict): Configuración
        history (dict): Historial entrenamiento
        model_state (dict): Estado del modelo
        **kwargs: Otros objetos
    
    Retorna:
        str: Ruta del archivo guardado
    
    Ejemplo:
        save_experiment('test_run', metrics={'acc': 0.9}, config={'lr': 0.001})
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)
    
    os.makedirs(save_dir, exist_ok=True)
    
    experiment_data = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'metrics': metrics,
        'config': config,
        'history': history,
        'model_state': model_state,
        **kwargs
    }
    
    save_object(experiment_data, filepath)
    return filepath


def list_experiments(save_dir='experiments', pattern='*.pkl'):
    """
    Lista experimentos guardados
    
    Parámetros:
        save_dir (str): Directorio
        pattern (str): Patrón de archivos
    
    Retorna:
        list: Lista de archivos encontrados
    
    Ejemplo:
        exps = list_experiments('experiments')
    """
    import glob
    
    if not os.path.exists(save_dir):
        print(f"El directorio '{save_dir}' no existe.")
        return []
    
    search_pattern = os.path.join(save_dir, pattern)
    files = sorted(glob.glob(search_pattern), reverse=True)
    
    print(f"\nExperimentos encontrados en '{save_dir}':")
    print("=" * 70)
    
    for i, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath) / 1024
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        
        print(f"{i}. {filename}")
        print(f"   Tamaño: {file_size:.2f} KB | Modificado: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("=" * 70)
    return files


def autoargs(func):
    """
    Decorador para guardar automáticamente los argumentos del constructor
    
    Ejemplo:
        @autoargs
        def __init__(self, in_features, out_features):
            pass
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        sig = inspect.signature(func)
        params = sig.parameters
        param_names = list(params.keys())[1:]
        
        self._init_args = {}
        
        for i, arg in enumerate(args):
            if i < len(param_names):
                self._init_args[param_names[i]] = arg
        
        self._init_args.update(kwargs)
        
        return func(self, *args, **kwargs)
    
    return wrapper


def save_model_auto(model, filename, optimizer=None, scheduler=None, epoch=None):
    """
    Guarda modelo con código fuente y estado completo usando dill
    
    Parámetros:
        model: Modelo PyTorch
        filename (str): Ruta del archivo
        optimizer: Optimizador (opcional)
        scheduler: Scheduler (opcional)
        epoch (int): Época actual (opcional)
    
    Retorna:
        None
    
    Ejemplo:
        save_model_auto(model, 'model.pt', optimizer=opt, epoch=10)
    """
    import torch
    import dill
    import inspect
    
    model_class = type(model)
    source_code = inspect.getsource(model_class)
    
    buffer = {
        "code": source_code,
        "class_name": model_class.__name__,
        "state_dict": model.state_dict(),
        "args_info": getattr(model, "_init_args", {})
    }
    
    if optimizer is not None:
        buffer["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        buffer["scheduler_state_dict"] = scheduler.state_dict()
    
    if epoch is not None:
        buffer["epoch"] = epoch
    
    torch.save(buffer, filename, pickle_module=dill)
    print(f"Modelo guardado en: {filename}")


def load_model_auto(filename, device="cpu"):
    """
    Carga modelo guardado con save_model_auto
    
    Parámetros:
        filename (str): Ruta del archivo
        device (str): 'cpu' o 'cuda'
    
    Retorna:
        tuple: (model, metadata) o (model, optimizer, scheduler, epoch, metadata)
    
    Ejemplo:
        model, meta = load_model_auto('model.pt')
        # o con optimizador:
        model, opt, sched, epoch, meta = load_model_auto('model.pt')
    """
    import torch
    import torch.nn as nn
    buffer = torch.load(filename, map_location=device, pickle_module=dill)
    
    module = types.ModuleType("loaded_model_module")
    module.__dict__.update({
        "torch": torch,
        "nn": nn,
        "F": torch.nn.functional,
        "autoargs": autoargs
    })
    
    exec(buffer["code"], module.__dict__)
    cls = getattr(module, buffer["class_name"])
    
    try:
        model = cls(**buffer["args_info"])
    except TypeError:
        model = cls()
    
    model.load_state_dict(buffer["state_dict"])
    model.to(device)
    
    optimizer = None
    if "optimizer_state_dict" in buffer:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(buffer["optimizer_state_dict"])
    
    scheduler = None
    if "scheduler_state_dict" in buffer and optimizer is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scheduler.load_state_dict(buffer["scheduler_state_dict"])
    
    epoch = buffer.get("epoch", None)
    
    metadata = {
        "code": buffer["code"],
        "class_name": buffer["class_name"],
        "args_info": buffer["args_info"]
    }
    
    if optimizer is None and scheduler is None and epoch is None:
        return model, metadata
    else:
        return model, optimizer, scheduler, epoch, metadata


def extract_data_from_loader(dataloader):
    """
    Extrae todos los datos de un DataLoader
    
    Parámetros:
        dataloader: PyTorch DataLoader
    
    Retorna:
        tuple: (X, y) o (X, None) si no hay labels
    
    Ejemplo:
        X, y = extract_data_from_loader(train_loader)
    """
    X_list = []
    y_list = []
    
    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            X_batch, y_batch = batch
            X_list.append(X_batch)
            y_list.append(y_batch)
        else:
            X_list.append(batch)
    
    X = torch.cat(X_list, dim=0)
    
    if y_list:
        y = torch.cat(y_list, dim=0)
        return X, y
    else:
        return X, None


def recover_activation_kwargs(activation_fn, activation_type):
    """
    Recupera kwargs de una función de activación instanciada
    
    Parámetros:
        activation_fn: Objeto de función de activación
        activation_type (str): Tipo ('polynomial', 'bspline', etc.)
    
    Retorna:
        dict: Kwargs recuperados
    
    Ejemplo:
        kwargs = recover_activation_kwargs(layer.input_activation_fn, 'polynomial')
    """
    kwargs = {}
    
    if activation_fn is None:
        return kwargs

    # Polynomial
    if activation_type == "polynomial":
        if hasattr(activation_fn, 'degree'):
            kwargs['degree'] = activation_fn.degree
            
    # B-Splines
    elif activation_type in ["bspline", "fbspline", "fast_bspline"]:
        if hasattr(activation_fn, 'grid_size'):
            kwargs['grid'] = activation_fn.grid_size
        if hasattr(activation_fn, 'spline_order'):
            kwargs['k'] = activation_fn.spline_order
        if hasattr(activation_fn, 'grid_min'):
            kwargs['grid_min'] = activation_fn.grid_min
        if hasattr(activation_fn, 'grid_max'):
            kwargs['grid_max'] = activation_fn.grid_max
            
    # RBF
    elif activation_type == "rbf":
        if hasattr(activation_fn, 'n_centers'):
            kwargs['n_centers'] = activation_fn.n_centers
        if hasattr(activation_fn, 'centers'):
            kwargs['centers'] = activation_fn.centers
        if hasattr(activation_fn, 'sigma'):
            kwargs['sigma'] = activation_fn.sigma
    
    # Legendre
    elif activation_type == "legendre":
        if hasattr(activation_fn, 'degree'):
            kwargs['degree'] = activation_fn.degree
    
    # Chebyshev
    elif activation_type == "chebyshev":
        if hasattr(activation_fn, 'degree'):
            kwargs['degree'] = activation_fn.degree
        if hasattr(activation_fn, 'kind'):
            kwargs['kind'] = activation_fn.kind
    
    # Gegenbauer
    elif activation_type == "gegenbauer":
        if hasattr(activation_fn, 'degree'):
            kwargs['degree'] = activation_fn.degree
        if hasattr(activation_fn, 'alpha'):
            kwargs['alpha'] = activation_fn.alpha
    
    # Jacobi
    elif activation_type == "jacobi":
        if hasattr(activation_fn, 'degree'):
            kwargs['degree'] = activation_fn.degree
        if hasattr(activation_fn, 'alpha'):
            kwargs['alpha'] = activation_fn.alpha
        if hasattr(activation_fn, 'beta'):
            kwargs['beta'] = activation_fn.beta
    
    # Sine
    elif activation_type == "sine":
        if hasattr(activation_fn, 'n_frequencies'):
            kwargs['n_frequencies'] = activation_fn.n_frequencies
        if hasattr(activation_fn, 'omega'):
            kwargs['omega'] = activation_fn.omega
    
    # Rational
    elif activation_type == "rational":
        if hasattr(activation_fn, 'num_degree'):
            kwargs['num_degree'] = activation_fn.num_degree
        if hasattr(activation_fn, 'den_degree'):
            kwargs['den_degree'] = activation_fn.den_degree
    
    # Wavelet
    elif activation_type == "wavelet":
        if hasattr(activation_fn, 'n_wavelets'):
            kwargs['n_wavelets'] = activation_fn.n_wavelets
        if hasattr(activation_fn, 'wavelet_type'):
            kwargs['wavelet_type'] = activation_fn.wavelet_type
        if hasattr(activation_fn, 'scale'):
            kwargs['scale'] = activation_fn.scale

    return kwargs


def extract_kan_config(kan_network):
    """
    Extrae la configuración completa de una red DualFlexKAN
    
    Parámetros:
        kan_network: Instancia de DualFlexKAN
    
    Retorna:
        dict: Configuración completa con todas las estrategias y parámetros
    
    Ejemplo:
        config = extract_kan_config(model.encoder.network)
    """
    config = {
        'input_function_strategies': [],
        'input_activation_types': [],
        'input_activation_kwargs': [],
        'output_function_strategies': [],
        'output_activation_types': [],
        'output_activation_kwargs': [],
        'dropout_probs': [],
        'dropout_positions': [],
        'use_batch_norms': [],
        'batch_norm_positions': [],
        'regularization_orders': [],
        'batch_norm_momentums': [],
        'batch_norm_epsilons': []
    }
    
    for layer in kan_network.layers:
        # Estrategias y tipos
        config['input_function_strategies'].append(layer.input_function_strategy)
        config['input_activation_types'].append(layer.input_activation_type)
        config['output_function_strategies'].append(layer.output_function_strategy)
        config['output_activation_types'].append(layer.output_activation_type)
        
        # Kwargs de activaciones
        in_kwargs = recover_activation_kwargs(layer.input_activation_fn, layer.input_activation_type)
        out_kwargs = recover_activation_kwargs(layer.output_activation_fn, layer.output_activation_type)
        config['input_activation_kwargs'].append(in_kwargs)
        config['output_activation_kwargs'].append(out_kwargs)
        
        # Dropout
        config['dropout_probs'].append(layer.dropout_prob)
        config['dropout_positions'].append(layer.dropout_position)
        
        # Batch Norm
        config['use_batch_norms'].append(layer.use_batch_norm)
        config['batch_norm_positions'].append(layer.batch_norm_position)
        
        # Regularization order
        config['regularization_orders'].append(layer.regularization_order)
        
        # Batch norm params
        if layer.batch_norm_before is not None:
            config['batch_norm_momentums'].append(layer.batch_norm_before.momentum)
            config['batch_norm_epsilons'].append(layer.batch_norm_before.eps)
        elif layer.batch_norm_after is not None:
            config['batch_norm_momentums'].append(layer.batch_norm_after.momentum)
            config['batch_norm_epsilons'].append(layer.batch_norm_after.eps)
        else:
            config['batch_norm_momentums'].append(0.1)
            config['batch_norm_epsilons'].append(1e-5)
        
    return config
