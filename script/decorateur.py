### Script avec les décorateurs utilisés ###

import time
import functools

# decorateur pour mesurer le temps d'execution d'une fonction
def compute_time(function):
    def somefunction(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        print(f"Temps d'exécution : {end_time - start_time:.0f} secondes")
        return result
    return somefunction


def log_execution_time(filename):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()  # Temps avant l'exécution de la fonction
            
            # Exécution de la fonction décorée
            result = func(*args, **kwargs)
            
            end_time = time.time()    # Temps après l'exécution de la fonction
            
            # Calcul du temps d'exécution
            execution_time = end_time - start_time
            
            # Écriture des détails dans le fichier
            with open(filename, 'a') as file:
                # Écriture des paramètres et du temps d'exécution
                file.write(f"Function {func.__name__} called with arguments {args} and keyword arguments {kwargs}.\n")
                file.write(f"Execution time: {execution_time:.0f} seconds.\n\n")
            
            return result
        
        return wrapper
    return decorator