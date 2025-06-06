# Importamos el entorno de gymnasium para control clásico (CartPole)
import gymnasium as gym  
# Importamos utilidad de juego (no usada en este ejemplo pero se incluye)
from gymnasium.utils.play import play  
# Para manejar retardos en tiempo real y sincronizar la lectura/visualización
import time  
# Biblioteca fundamental para cálculos numéricos y estadísticos
import numpy as np  
# BrainFlow: gestión de la placa OpenBCI (conexión, streaming, canales)
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds  
# BrainFlow: filtros y procesamiento de señales EEG
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowOperations, DetrendOperations  
# Matplotlib: para graficar en tiempo real las métricas de EEG
import matplotlib.pyplot as plt  
# deque: doble cola con tamaño máximo para buffer circular de métricas
from collections import deque  

# Parámetro ID de la placa Cyton según BrainFlow
BOARD_ID = BoardIds.CYTON_BOARD.value  
# Puerto serie donde está conectada la placa (ajustar a tu sistema)
SERIAL_PORT = 'COM6'  
# Índice del canal EEG que usaremos para detectar “pensar en izquierda”
ELECTRODE_CHANNEL_LEFT = 1  
# Índice del canal EEG que usaremos para detectar “pensar en derecha”
ELECTRODE_CHANNEL_RIGHT = 2  
# Frecuencia de muestreo de la placa (Hz), obtenida dinámicamente
SAMPLING_RATE = BoardShim.get_sampling_rate(BOARD_ID)  
# Tamaño del buffer interno de datos (en muestras): 5 s de datos
BUFFER_SIZE = SAMPLING_RATE * 5  

# Duración de cada fase de calibración (segundos)
CALIBRATION_DURATION_PER_ACTION = 10  
# Número de fases de calibración (izquierda, derecha)
N_ACTIONS_CALIBRATION = 2  # Izquierda, Derecha (podrías añadir "reposo" o "nada")

# Parámetros del juego
ENV_NAME = 'MountainCar-v0'  # CAMBIADO DE CartPole-v1


class BCIController:
    def __init__(self, board_id, serial_port):
        # Guardamos ID y puerto
        self.board_id = board_id  
        self.serial_port = serial_port  
        # Preparamos parámetros de conexión BrainFlow
        self.params = BrainFlowInputParams()  
        self.params.serial_port = self.serial_port  
        # Inicializamos la sesión BrainFlow con esos parámetros
        self.board = BoardShim(self.board_id, self.params)  
        # Obtenemos la lista de canales EEG disponibles
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)  
        # Volvemos a leer la tasa de muestreo (Hz)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)  
        # Umbrales que calcularemos en calibración
        self.threshold_left = 0  
        self.threshold_right = 0  

    def connect(self):
        try:
            # Preparamos la sesión y arrancamos el streaming de datos
            self.board.prepare_session()  
            self.board.start_stream(BUFFER_SIZE)  
            print("OpenBCI Cyton conectada y transmitiendo.")  
            return True  # Conexión exitosa
        except Exception as e:
            print(f"Error al conectar con OpenBCI: {e}")  
            return False  # Falla al conectar

    def get_eeg_data(self, num_samples=None):
        # Si no indicamos muestra, devolvemos todo el buffer interno
        if num_samples is None:
            return self.board.get_board_data()  
        else:
            # Si indicamos número, devolvemos solo las últimas `num_samples`
            return self.board.get_current_board_data(num_samples)  

    def disconnect(self):
        # Si la sesión está activa, detenemos y liberamos
        if self.board.is_prepared():
            self.board.stop_stream()  
            self.board.release_session()  
            print("OpenBCI Cyton desconectada.")  

    def _preprocess_data(self, data, channel_index):
        """Calcula la potencia en la banda alfa (8-12 Hz) para el canal especificado."""
        # Extraemos la señal del canal indicado
        channel_data = data[channel_index]  
        # Quitamos tendencia constante
        DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)  

        current_data_len = len(channel_data)

        # Si no hay suficientes datos, no se puede calcular PSD de forma fiable.
        # Un mínimo práctico para nfft podría ser 4. Si current_data_len es menor,
        # nfft probablemente será demasiado pequeño.
        # Para Cyton (250Hz) pidiendo 1s de datos, current_data_len debería ser 250.
        if current_data_len < 4: # Umbral mínimo de puntos para PSD
            print(f"Advertencia: Datos insuficientes (longitud: {current_data_len}) para calcular PSD en canal {channel_index}. Retornando 0.0.")
            return 0.0

        # Calculamos nfft basado en la longitud actual de los datos.
        nfft = DataFilter.get_nearest_power_of_two(current_data_len)
        
        # Aseguramos que nfft <= current_data_len, como requiere get_psd_welch.
        # Si get_nearest_power_of_two redondeó hacia arriba más allá de current_data_len
        # (ej. current_data_len=250 -> nfft=256), lo reducimos a la potencia de dos inferior.
        if nfft > current_data_len:
            nfft //= 2
        
        # Si nfft se vuelve demasiado pequeño después de la posible división (ej. < 4),
        # la PSD puede no ser fiable o causar error.
        # Este chequeo es una salvaguarda adicional.
        if nfft < 4: 
            print(f"Advertencia: nfft ({nfft}) es demasiado pequeño tras el ajuste (longitud de datos: {current_data_len}) para calcular PSD en canal {channel_index}. Retornando 0.0.")
            return 0.0
        
        # psd_data[0] son las amplitudes, psd_data[1] son las frecuencias
        psd_data = DataFilter.get_psd_welch(
            channel_data,
            nfft, # Número de puntos FFT
            nfft // 2, # Superposición entre segmentos (típicamente 50%)
            self.sampling_rate, # Frecuencia de muestreo original de la señal
            WindowOperations.HANNING.value # Tipo de ventana
        )

        # Calculamos la potencia en la banda alfa (8-12 Hz) a partir de la PSD
        band_power_alpha = DataFilter.get_band_power(
            psd_data,
            8.0,  # frecuencia mínima de la banda alfa
            12.0  # frecuencia máxima de la banda alfa
        )
        
        return band_power_alpha

    def calibrate(self):
        print("\n--- Fase de Calibración ---")
        # Etiquetas de acción para cada fase
        actions = ["izquierda", "derecha"]  
        action_metrics_left = []   # Almacena métricas de canal izquierdo
        action_metrics_right = []  # Almacena métricas de canal derecho

        # Repetimos N_ACTIONS_CALIBRATION veces (para izquierda y derecha)
        for i in range(N_ACTIONS_CALIBRATION):
            # Elegimos la etiqueta según el índice
            current_action_label = actions[i % len(actions)]
            print(f"Enfócate en {current_action_label.upper()} durante {CALIBRATION_DURATION_PER_ACTION}s.")
            time.sleep(2)  # breve espera antes de grabar

            start_time = time.time()
            temp_metrics_left = []  # métricas de esta fase (izquierda)
            temp_metrics_right = [] # métricas de esta fase (derecha)

            # Mientras no se cumpla el tiempo de calibración
            while time.time() - start_time < CALIBRATION_DURATION_PER_ACTION:
                # Obtenemos 1 s de datos
                data = self.get_eeg_data(self.sampling_rate)
                if data.shape[1] > 0:
                    # Procesamos cada canal y guardamos la métrica
                    ml = self._preprocess_data(data, ELECTRODE_CHANNEL_LEFT)
                    mr = self._preprocess_data(data, ELECTRODE_CHANNEL_RIGHT)
                    temp_metrics_left.append(ml)
                    temp_metrics_right.append(mr)
                    print(f"  Izquierda: {ml:.2f}, Derecha: {mr:.2f}")
                time.sleep(1)  # esperar 1 s antes de siguiente muestra

            # Acumulamos las métricas en la lista global según la acción
            if current_action_label == "izquierda":
                action_metrics_left.extend(temp_metrics_left)
            else:
                action_metrics_right.extend(temp_metrics_right)

        # Si no hay suficientes datos en alguna fase, ponemos umbrales por defecto
        if not action_metrics_left or not action_metrics_right:
            print("Datos insuficientes para calibrar. Se usan umbrales por defecto.")
            self.threshold_left = 5
            self.threshold_right = 5
            return

        # Calculamos umbrales: media + 0.5 × desviación estándar
        self.threshold_left = np.mean(action_metrics_left) + 0.5 * np.std(action_metrics_left)
        self.threshold_right = np.mean(action_metrics_right) + 0.5 * np.std(action_metrics_right)

        print("Calibración completada.")
        print(f"  Umbral Izquierda: {self.threshold_left:.2f}")
        print(f"  Umbral Derecha:   {self.threshold_right:.2f}")
        print("-------------------------")

    def get_action_from_eeg(self):
        """Lee 1 s de EEG, procesa métricas y decide acción para MountainCar (0:Izq, 1:Nada, 2:Der)."""
        data = self.get_eeg_data(self.sampling_rate)
        # Si no hay datos suficientes, devolvemos None para métricas y acción por defecto (1: no acelerar)
        if data.shape[1] == 0:
            return 1, None, None # Acción por defecto: no acelerar, sin métricas

        # Calculamos métricas de ambos canales
        ml = self._preprocess_data(data, ELECTRODE_CHANNEL_LEFT)
        mr = self._preprocess_data(data, ELECTRODE_CHANNEL_RIGHT)
        print(f"EEG real – Izq:{ml:.2f} (thr {self.threshold_left:.2f}), Der:{mr:.2f} (thr {self.threshold_right:.2f})")

        diff_threshold = 1.0  # diferencia mínima para tomar la decisión
        action = 1 # Acción por defecto para MountainCar: no acelerar

        # Si la señal izquierda excede su umbral y supera a la derecha
        if ml > self.threshold_left and ml > mr + diff_threshold:
            print("Acción: IZQUIERDA (0)")
            action = 0 # Empujar a la izquierda para MountainCar
        # Si la señal derecha excede su umbral y supera a la izquierda
        elif mr > self.threshold_right and mr > ml + diff_threshold:
            print("Acción: DERECHA (2)")
            action = 2 # Empujar a la derecha para MountainCar
        else:
            print("Acción: NINGUNA (1)") # No acelerar

        # Devolvemos la acción y las métricas para graficar
        return action, ml, mr


def main():
    # Instanciamos el controlador BCI con ID y puerto
    bci = BCIController(BOARD_ID, SERIAL_PORT)  # SERIAL_PORT es 'COM6' en el archivo del usuario
    if not bci.connect():
        return  # salimos si no conectamos

    # --- Fase de Calibración Opcional ---
    calibrate_choice = input("¿Deseas realizar la calibración de umbrales? (s/n, por defecto n): ").lower()
    if calibrate_choice == 's':
        bci.calibrate()
        # Verificar si la calibración fue efectiva (si los umbrales son > 0 o algún valor esperado)
        if bci.threshold_left == 0 and bci.threshold_right == 0: # Ejemplo de chequeo básico
            print("Advertencia: La calibración podría no haber establecido umbrales significativos (siguen en 0).")
            print("Se usarán umbrales por defecto (5.0) para esta sesión si la calibración no los modificó.")
            # Establecer valores por defecto si la calibración no los cambió de 0
            if bci.threshold_left == 0: bci.threshold_left = 5.0
            if bci.threshold_right == 0: bci.threshold_right = 5.0
    else:
        print("Calibración omitida.")
        # Establecer umbrales por defecto si no se calibró y no hay valores previos (ej. primera ejecución o si eran 0)
        if bci.threshold_left == 0 and bci.threshold_right == 0:
            bci.threshold_left = 5.0  # Valor por defecto razonable
            bci.threshold_right = 5.0 # Valor por defecto razonable
            print("Usando umbrales por defecto (5.0).")
        else:
            print("Usando umbrales previamente establecidos o calibrados.")
    
    print(f"  Umbral Izquierda inicial: {bci.threshold_left:.2f}")
    print(f"  Umbral Derecha inicial:   {bci.threshold_right:.2f}")

    plt.ion() # Modo interactivo de Matplotlib ON para actualizaciones dinámicas
    fig = None # Inicializar fig para poder cerrarla después si es necesario

    # --- Bucle Principal de Juego (para jugar múltiples partidas) ---
    while True:
        print("\\n--- Iniciando Nueva Partida de CartPole ---")
        print("Controla el carro con tu actividad cerebral.")
        print("Cierra la ventana del juego para terminar la partida actual (o presiona Ctrl+C en la terminal).")

        env = gym.make(ENV_NAME, render_mode='human')
        _ = env.reset() # Reiniciar el entorno para la nueva partida, la observación no se usa inmediatamente aquí
        
        # --- Inicialización de la gráfica en tiempo real para la partida actual ---
        buf_len = 100  # longitud del buffer (número de muestras para mostrar)
        left_buf = deque(maxlen=buf_len)   # buffer circular para métricas del canal izquierdo
        right_buf = deque(maxlen=buf_len)  # buffer circular para métricas del canal derecho
        
        # Cerrar figura anterior si existe, para no superponer o causar errores
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6)) # Crear nueva figura y ejes
        fig.suptitle("Métricas EEG en Tiempo Real", fontsize=14)

        # Usar los umbrales actuales (calibrados o por defecto) para la gráfica
        current_threshold_left = bci.threshold_left
        current_threshold_right = bci.threshold_right

        line1, = ax1.plot([], [], color='dodgerblue', linestyle='-', linewidth=2)
        ax1.set_title(f'Actividad Cerebral - Izquierda (Umbral: {current_threshold_left:.2f})', fontsize=10)
        ax1.set_ylabel('Métrica EEG', fontsize=9)
        ax1.axhline(y=current_threshold_left, color='lightcoral', linestyle='--', linewidth=1.5, label=f'Umbral Izq ({current_threshold_left:.2f})')
        ax1.set_ylim(0, max(current_threshold_left * 2.5, 2.0)) # Límite Y inicial un poco más generoso
        ax1.set_xlim(0, buf_len)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, linestyle=':', alpha=0.7)

        line2, = ax2.plot([], [], color='orangered', linestyle='-', linewidth=2)
        ax2.set_title(f'Actividad Cerebral - Derecha (Umbral: {current_threshold_right:.2f})', fontsize=10)
        ax2.set_xlabel('Muestras de Tiempo (~1 muestra/seg)', fontsize=9)
        ax2.set_ylabel('Métrica EEG', fontsize=9)
        ax2.axhline(y=current_threshold_right, color='lightcoral', linestyle='--', linewidth=1.5, label=f'Umbral Der ({current_threshold_right:.2f})')
        ax2.set_ylim(0, max(current_threshold_right * 2.5, 2.0)) 
        ax2.set_xlim(0, buf_len)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, linestyle=':', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar layout para que el suptitle y etiquetas no se solapen
        fig.canvas.draw_idle() # Dibujar la figura inicialmente
        plt.show(block=False) # Mostrar la figura sin bloquear el flujo del programa

        game_active = True # Flag para controlar el bucle de la partida actual
        try:
            terminated, truncated = False, False
            total_reward = 0
            last_action = 1  # acción por defecto para MountainCar: no acelerar

            # Bucle de una partida individual
            while game_active and not (terminated or truncated):
                # Verificar si la ventana de la gráfica sigue abierta
                if not plt.fignum_exists(fig.number): 
                    print("Ventana de gráfica cerrada por el usuario. Terminando partida.")
                    break # Salir del bucle de esta partida

                # Leemos acción y métricas de EEG
                action, ml, mr = bci.get_action_from_eeg() # action es 0, 1, o 2 para MountainCar
                
                # Actualizamos buffers para graficar
                if ml is not None: # Si hay datos válidos
                    left_buf.append(ml)
                    right_buf.append(mr)
                else: # Si no hay datos (ej. al inicio o error de lectura), añadir NaN para no graficar un 0
                    left_buf.append(np.nan) 
                    right_buf.append(np.nan)
                
                # Dibujamos la curva actualizada
                line1.set_data(range(len(left_buf)), list(left_buf))
                line2.set_data(range(len(right_buf)), list(right_buf))

                # Auto-ajuste dinámico de los límites Y de las gráficas si es necesario
                if ml is not None and not np.isnan(ml): # Solo ajustar si ml es un número válido
                    current_max_y_left = ax1.get_ylim()[1]
                    if ml > current_max_y_left * 0.90: # Si la métrica se acerca al 90% del borde superior
                        ax1.set_ylim(0, ml * 1.2 + 1) # Aumentar el límite superior con un margen
                
                if mr is not None and not np.isnan(mr): # Solo ajustar si mr es un número válido
                    current_max_y_right = ax2.get_ylim()[1]
                    if mr > current_max_y_right * 0.90:
                        ax2.set_ylim(0, mr * 1.2 + 1)
                
                # Determinar la acción actual que se enviará al entorno
                current_action = action # Para MountainCar, 'action' ya es 0, 1 o 2.
                # No es necesario el manejo especial de last_action a menos que haya un caso donde no se defina acción.
                # current_action = action # La acción ya es 0, 1 o 2.

                # Actualizar el título de la figura con la acción actual
                action_str = "ESPERANDO..." # Texto por defecto si algo sale mal
                if current_action == 0:
                    action_str = "IZQUIERDA"
                elif current_action == 1:
                    action_str = "NINGUNA"
                elif current_action == 2:
                    action_str = "DERECHA"
                
                fig.suptitle(f"Métricas EEG en Tiempo Real - Acción: {action_str}", fontsize=14)
                
                try:
                    fig.canvas.draw_idle() # Programar redibujado de forma eficiente
                    fig.canvas.flush_events() # Procesar eventos de la GUI (importante para que la gráfica se actualice)
                except Exception as e_draw: 
                    if not plt.fignum_exists(fig.number): # Si el error es porque la ventana se cerró
                        print(f"Error al dibujar (la ventana de gráfica ya no existe): {e_draw}")
                        break # Terminar la partida si la gráfica se cerró
                    else: # Otro tipo de error de dibujado
                        print(f"Error inesperado al dibujar la gráfica: {e_draw}")
                        # Considerar si continuar o no; por ahora, se imprime y continúa
                
                # Si la BCI define una acción, la usamos; si no, repetimos la anterior (que ya incluye el caso de "no acción clara" -> 1)
                # La función get_action_from_eeg ya devuelve una acción válida (0, 1, o 2)
                current_action = action # La acción ya es 0, 1 o 2.
                # No necesitamos last_action de la misma manera que antes si get_action_from_eeg siempre da una acción.
                # Sin embargo, si get_action_from_eeg pudiera fallar y devolver None para la acción, necesitaríamos last_action.
                # Por ahora, get_action_from_eeg siempre devuelve 0, 1, o 2.
                # last_action = current_action # Opcional, si queremos rastrearla.

                # Renderizamos el entorno y damos un paso con la acción
                if env.render_mode == 'human':
                    try:
                        env.render() # Renderizar si el modo es 'human'
                    except Exception as e_render: # Capturar errores si la ventana de Gym se cierra
                        print(f"Error al renderizar el entorno del juego (¿ventana cerrada?): {e_render}")
                        break # Terminar la partida si no se puede renderizar

                try:
                    obs, rew, terminated, truncated, info = env.step(current_action)
                except Exception as e_step: # Por si el entorno de Gym tiene problemas (ej. ventana cerrada)
                    print(f"Error durante env.step(): {e_step}")
                    break # Terminar la partida
                total_reward += rew

                # Si el episodio termina, mostramos puntuación
                if terminated or truncated:
                    print(f"Partida terminada. Puntuación: {total_reward}")
                    break # Salir del bucle de esta partida

                time.sleep(0.05)  # Pausa para no saturar CPU y dar tiempo a la BCI/gráfica

        except KeyboardInterrupt:
            print("\\nPartida interrumpida por el usuario (Ctrl+C).")
        finally:
            # Limpieza al final de cada partida
            print("Cerrando entorno de la partida actual...")
            env.close() # Cerramos el entorno del juego actual

        # Preguntar si quiere jugar otra partida
        play_again_choice = input("¿Quieres jugar otra partida? (s/n, por defecto s): ").lower()
        if play_again_choice == 'n':
            break # Salir del bucle principal de juego (while True)

    # --- Limpieza Final del Programa ---
    if fig is not None and plt.fignum_exists(fig.number):
        plt.close(fig) # Cerrar la ventana de la gráfica si aún está abierta
    
    bci.disconnect() # Desconectamos la BCI al final de todas las partidas
    print("Programa finalizado.")


if __name__ == '__main__':
    main()