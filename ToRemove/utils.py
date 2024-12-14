import time

def log_event(event):
    """Log an event with a timestamp."""
    with open('system.log', 'a') as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {event}\n")

def calculate_accuracy(total_matches, total_attempts):
    """Calcula a acurácia baseada em correspondências e tentativas."""
    if total_attempts == 0:
        return 0.0
    return (total_matches / total_attempts) * 100


def time_execution(func):
    """Decorator to measure and log the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        log_event(f"Executed {func.__name__} in {execution_time:.4f} seconds")
        return result, execution_time
    return wrapper
