import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# from PIL import Image, ImageTk # 不要
import threading
import subprocess
import platform
import sys
import math
import multiprocessing
from queue import Queue, Empty
import time

# --- グローバル変数の定義 ---
input_directory = ""
output_directory = ""
processing_queue = None
stop_requested = False
pause_event = None
processing_pool = None
start_time = 0.0
file_count = 0
processed_count = 0
error_count = 0
is_processing = False
is_paused = False
# -------------------------


def get_grid_tuple(grid_layout_str):
    try:
        count_str = grid_layout_str.split(" ")[0]
        count = int(count_str)
        side = int(math.sqrt(count))
        if side * side == count:
            return (side, side)
        else:
            print(
                f"警告: 不正グリッドレイアウト指定 {grid_layout_str}。デフォルト4x4使用。"
            )
            return (4, 4)
    except Exception:
        print(
            f"警告: グリッドレイアウト解析失敗 {grid_layout_str}。デフォルト4x4使用。"
        )
        return (4, 4)


def create_thumbnail_grid(
    video_path,
    output_dir,
    grid_layout="16 (4x4)",
    output_grid_size="1920x1080",
    use_gpu=False,
    pause_event_worker=None,
):
    original_basename = os.path.basename(video_path)
    ocl_active = False
    try:
        if use_gpu:
            try:
                if cv2.ocl.haveOpenCL():
                    cv2.ocl.setUseOpenCL(True)
                    if cv2.ocl.useOpenCL():
                        ocl_active = True
                else:
                    use_gpu = False
            except Exception as e_ocl:
                print(f"PID {os.getpid()}: Failed to set OpenCL: {e_ocl}")
                use_gpu = False
        else:
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass

        grid_size = get_grid_tuple(grid_layout)
        try:
            grid_width_str, grid_height_str = output_grid_size.split("x")
            target_grid_width, target_grid_height = int(grid_width_str), int(
                grid_height_str
            )
            if target_grid_width <= 0 or target_grid_height <= 0:
                raise ValueError("グリッド幅/高さは正の値")
        except ValueError as e:
            raise ValueError(f"無効なグリッドサイズ形式: {e}")
        thumbnail_width, thumbnail_height = (
            target_grid_width // grid_size[1],
            target_grid_height // grid_size[0],
        )
        if thumbnail_width <= 0 or thumbnail_height <= 0:
            raise ValueError("計算サムネイルサイズ不正(≦0)")
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        # --- エラーメッセージ修正 ---
        if not cap.isOpened():
            raise Exception(
                f"動画を開けません (コーデック/オーディオビットレート問題の可能性あり)"
            )
        # -----------------------
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise Exception(f"フレーム数0または取得失敗")
        num_thumbnails_target = grid_size[0] * grid_size[1]
        if frame_count < num_thumbnails_target:
            frame_interval, num_actual_thumbnails = 1, frame_count
        elif num_thumbnails_target > 0:
            frame_interval, num_actual_thumbnails = (
                frame_count // num_thumbnails_target,
                num_thumbnails_target,
            )
        else:
            frame_interval, num_actual_thumbnails = 1, 0
        thumbnails = []
        for i in range(num_actual_thumbnails):
            if pause_event_worker:
                pause_event_worker.wait()
            frame_number = i * frame_interval
            frame_number = min(frame_number, frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue
            try:
                resized_frame = cv2.resize(frame, (thumbnail_width, thumbnail_height))
                thumbnails.append(resized_frame)
            except cv2.error:
                continue
        cap.release()
        if not thumbnails:
            raise Exception("有効なフレーム無し")
        actual_grid_width, actual_grid_height = (
            grid_size[1] * thumbnail_width,
            grid_size[0] * thumbnail_height,
        )
        grid = np.zeros((actual_grid_height, actual_grid_width, 3), dtype=np.uint8)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                thumbnail_index = i * grid_size[1] + j
                if thumbnail_index < len(thumbnails):
                    x, y = j * thumbnail_width, i * thumbnail_height
                    try:
                        grid[y : y + thumbnail_height, x : x + thumbnail_width] = (
                            thumbnails[thumbnail_index]
                        )
                    except ValueError:
                        continue
        filename_base = os.path.splitext(original_basename)[0]
        output_filename = f"{filename_base}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        try:
            is_success, im_buf_arr = cv2.imencode(
                ".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            if is_success:
                with open(output_path, "wb") as f_output:
                    f_output.write(im_buf_arr.tobytes())
            else:
                raise Exception("imencode失敗")
        except Exception as encode_err:
            try:
                cv2.imwrite(output_path, grid)
            except Exception as imwrite_err:
                raise Exception(f"保存失敗: {encode_err},{imwrite_err}")
        return original_basename, "完了"
    except Exception as e:
        return original_basename, f"エラー: {e}"


# --- multiprocessing用ワーカ関数 ---
def worker_process_video(args):
    (
        video_path,
        output_dir,
        grid_layout,
        output_grid_size,
        use_gpu_flag,
        pause_event_ref,
        result_q,
    ) = args
    try:
        result = create_thumbnail_grid(
            video_path,
            output_dir,
            grid_layout,
            output_grid_size,
            use_gpu=use_gpu_flag,
            pause_event_worker=pause_event_ref,
        )
        if result_q:
            result_q.put(result)
    except Exception as e:
        if result_q:
            result_q.put((os.path.basename(video_path), f"予期せぬエラー: {e}"))


# --- GUI更新用関数 ---
def update_gui_from_queue():
    global processing_queue, processed_count, error_count, file_count, start_time, stop_requested, is_processing
    if stop_requested or not is_processing:
        return  # 中断・終了時は抜ける
    try:
        result_text.config(state=tk.NORMAL)
        items_processed_in_loop = 0
        while not processing_queue.empty():
            try:
                if stop_requested:
                    break
                result = processing_queue.get_nowait()
                items_processed_in_loop += 1
                processed_count += 1
                original_filename, status_msg = result
                if "エラー" in status_msg:
                    error_count += 1
                    result_text.tag_config("error", foreground="red")
                    result_text.insert(
                        tk.END, f"{original_filename}: {status_msg}\n", "error"
                    )
                else:
                    result_text.insert(tk.END, f"{original_filename}: {status_msg}\n")
            except Empty:
                break
            except Exception as e_get:
                print(f"キュー取得エラー: {e_get}")
                break
        if items_processed_in_loop > 0:
            result_text.see(tk.END)
        result_text.config(state=tk.DISABLED)
        if start_time > 0:
            elapsed_time = time.time() - start_time
            pause_status = "[一時停止中] " if is_paused else ""
            status_text = f"{pause_status}処理中: {processed_count}/{file_count} 完了 (エラー: {error_count}) | 経過: {elapsed_time:.1f}秒"
            status_label.config(text=status_text)
        if file_count > 0:
            progress_bar["value"] = processed_count
        if processed_count < file_count and not stop_requested:
            root.after(200, update_gui_from_queue)
        elif not stop_requested:
            is_processing = False
            finalize_processing()  # 正常完了
    except Exception as e:
        print(f"GUI更新ループエラー: {e}")
        status_label.config(text=f"GUI更新エラー: {e}")
        is_processing = False
        finalize_processing(error=True)


# --- 処理の後始末 ---
def finalize_processing(error=False, cancelled=False):
    global processing_pool, open_explorer_var, output_directory, start_time, processed_count, error_count, is_processing, is_paused, pause_event, stop_requested

    is_processing, is_paused = False, False
    start_button.config(state=tk.NORMAL)
    input_dir_button.config(state=tk.NORMAL)
    output_dir_button.config(state=tk.NORMAL)
    grid_layout_combobox.config(state="readonly")
    size_combobox.config(state="readonly")
    use_gpu_check.config(state=tk.NORMAL)
    process_count_combobox.config(state="readonly")  # プロセス数選択も有効化
    pause_resume_button.config(state=tk.DISABLED, text="一時停止")
    cancel_button.config(state=tk.DISABLED)
    if start_time > 0:
        total_time = time.time() - start_time
        if cancelled:
            final_status = f"処理中断: {processed_count}/{file_count} まで処理 (エラー: {error_count}) | 所要時間: {total_time:.1f} 秒"
        elif error:
            final_status = f"エラー発生: {processed_count}/{file_count} まで処理 (エラー: {error_count}) | 所要時間: {total_time:.1f} 秒"
        else:
            final_status = f"処理完了: {processed_count} ファイル (エラー: {error_count}) | 所要時間: {total_time:.1f} 秒"
        status_label.config(text=final_status)
        result_text.config(state=tk.NORMAL)
        result_text.insert(tk.END, f"\n{final_status}\n")
        result_text.see(tk.END)
        result_text.config(state=tk.DISABLED)
        start_time = 0.0
    elif error or cancelled:
        status_label.config(text="処理が中断されました")
    else:
        status_label.config(text="準備完了")
    result_text.config(state=tk.DISABLED)
    if processing_pool:
        try:
            if not processing_pool._closed:
                if not cancelled and not error:
                    processing_pool.close()
                    processing_pool.join(timeout=1)
                else:
                    print("プロセスプールを terminate します...")
                    processing_pool.terminate()
                    processing_pool.join(timeout=1)
        except Exception as e_pool:
            print(f"プロセスプールの終了中にエラー: {e_pool}")
        finally:
            processing_pool = None
    pause_event = None
    stop_requested = False
    if open_explorer_var.get() and not error and not cancelled:
        open_output_directory(output_directory)


# --- ディレクトリ処理 (メインスレッドから呼ばれる) ---
def process_directory_parallel(
    root_dir, out_dir, grid_layout_str, grid_size_str, use_gpu, process_count_str
):  # process_count_str追加
    global processing_queue, file_count, processed_count, error_count, processing_pool, start_time, pause_event, is_processing, is_paused, stop_requested

    file_count, processed_count, error_count = 0, 0, 0
    is_processing, is_paused, stop_requested = True, False, False
    video_files = []
    valid_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(valid_extensions):
                video_files.append(os.path.join(dirpath, filename))
                file_count += 1
    if file_count == 0:
        messagebox.showinfo("情報", "対象動画ファイル無し。")
        finalize_processing()
        return
    result_text.config(state=tk.NORMAL)
    result_text.delete("1.0", tk.END)
    manager = multiprocessing.Manager()
    processing_queue = manager.Queue()
    pause_event = manager.Event()
    pause_event.set()
    progress_bar["maximum"] = file_count
    progress_bar["value"] = 0
    status_label.config(text=f"処理準備中... 対象: {file_count} ファイル")
    result_text.insert(
        tk.END,
        f"処理開始... 対象: {file_count} ファイル (GPU: {'有効' if use_gpu else '無効'})\n",
    )
    root.update()

    # --- プロセス数決定ロジック ---
    if use_gpu:
        num_processes = 1
        print(f"GPU利用のためプロセス数を {num_processes} に強制します。")
        result_text.insert(
            tk.END, f"情報: GPU利用が有効なため、プロセス数を1に制限します。\n"
        )  # GUIにも表示
    elif process_count_str == "自動":
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        print(f"使用プロセス数 (自動): {num_processes}")
    else:
        try:
            num_processes = int(process_count_str)
            if num_processes < 1:
                num_processes = 1  # 1未満は1にする
            print(f"使用プロセス数 (指定): {num_processes}")
        except ValueError:
            num_processes = max(1, multiprocessing.cpu_count() - 1)  # 解析失敗時は自動
            print(f"プロセス数指定の解析失敗。自動設定({num_processes})を使用します。")
            result_text.insert(
                tk.END,
                f"警告: プロセス数指定の解析に失敗。自動設定({num_processes})を使用します。\n",
            )
    # --------------------------

    result_text.insert(
        tk.END, f"情報: {num_processes} プロセスで処理を開始します。\n"
    )  # 使用プロセス数をログに表示
    root.update()

    try:
        processing_pool = multiprocessing.Pool(processes=num_processes)
    except Exception as e_pool_create:
        messagebox.showerror("エラー", f"プロセスプール作成失敗:\n{e_pool_create}")
        finalize_processing(error=True)
        return
    tasks = [
        (
            path,
            out_dir,
            grid_layout_str,
            grid_size_str,
            use_gpu,
            pause_event,
            processing_queue,
        )
        for path in video_files
    ]
    try:
        processing_pool.map_async(worker_process_video, tasks)
    except Exception as e_map:
        messagebox.showerror("エラー", f"処理投入失敗:\n{e_map}")
        finalize_processing(error=True)
        return
    start_time = time.time()
    result_text.config(state=tk.DISABLED)
    root.after(100, update_gui_from_queue)


# --- browse系関数 ---
def browse_input_directory():
    global input_directory
    if is_processing:
        return
    dirname = filedialog.askdirectory()
    if dirname:
        input_directory = dirname
        input_dir_entry.delete(0, tk.END)
        input_dir_entry.insert(0, input_directory)


def browse_output_directory():
    global output_directory
    if is_processing:
        return
    dirname = filedialog.askdirectory()
    if dirname:
        output_directory = dirname
        output_dir_entry.delete(0, tk.END)
        output_dir_entry.insert(0, output_directory)


def open_output_directory(directory_to_open):
    if not directory_to_open or not os.path.isdir(directory_to_open):
        messagebox.showerror(
            "エラー", f"出力ディレクトリが見つかりません:\n{directory_to_open}"
        )
        return
    print(f"ディレクトリを開きます: {directory_to_open}")
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["explorer", os.path.normpath(directory_to_open)],
                capture_output=True,
                text=True,
                check=False,
            )
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["open", directory_to_open], capture_output=True, text=True, check=False
            )
        else:
            result = subprocess.run(
                ["xdg-open", directory_to_open],
                capture_output=True,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            if result.stderr:
                print(f"ディレクトリを開くコマンドのエラー出力:\n{result.stderr}")
            print(
                f"警告: ディレクトリを開くコマンド非ゼロ終了コード ({result.returncode})"
            )
    except FileNotFoundError:
        messagebox.showerror(
            "エラー", f"ファイルマネージャー起動コマンドが見つかりません。"
        )
    except Exception as e:
        messagebox.showerror("エラー", f"ディレクトリを開く際に予期せぬエラー: {e}")


# --- start_processing ---
def start_processing():
    global input_directory, output_directory, processing_pool, is_processing
    if is_processing:
        messagebox.showwarning("警告", "処理中です。")
        return
    current_input_dir = input_dir_entry.get()
    current_output_dir = output_dir_entry.get()
    selected_grid_layout = grid_layout_combobox.get()
    selected_output_size = size_combobox.get()
    use_gpu_flag = use_gpu_var.get()
    selected_process_count = process_count_combobox.get()  # プロセス数取得
    if not current_input_dir or not os.path.isdir(current_input_dir):
        messagebox.showerror("エラー", "有効な入力ディレクトリを選択。")
        return
    if not current_output_dir:
        messagebox.showerror("エラー", "出力ディレクトリを選択または入力。")
        return
    input_directory, output_directory = current_input_dir, current_output_dir
    start_button.config(state=tk.DISABLED)
    input_dir_button.config(state=tk.DISABLED)
    output_dir_button.config(state=tk.DISABLED)
    grid_layout_combobox.config(state=tk.DISABLED)
    size_combobox.config(state=tk.DISABLED)
    use_gpu_check.config(state=tk.DISABLED)
    process_count_combobox.config(state=tk.DISABLED)  # プロセス数選択も無効化
    pause_resume_button.config(state=tk.NORMAL, text="一時停止")
    cancel_button.config(state=tk.NORMAL)
    process_directory_parallel(
        input_directory,
        output_directory,
        selected_grid_layout,
        selected_output_size,
        use_gpu_flag,
        selected_process_count,
    )  # 引数追加


# --- pause_resume_processing, cancel_processing ---
def pause_resume_processing():
    global is_paused, pause_event, start_time, is_processing
    if not is_processing:
        return
    if is_paused:
        is_paused = False
        if pause_event:
            pause_event.set()
        pause_resume_button.config(text="一時停止")
        status_label.config(text=status_label.cget("text").replace("[一時停止中] ", ""))
        print("処理再開")
    else:
        is_paused = True
        if pause_event:
            pause_event.clear()
        pause_resume_button.config(text="再開")
        status_label.config(text="[一時停止中] " + status_label.cget("text"))
        print("処理一時停止")


def cancel_processing():
    global stop_requested, is_processing, processing_pool
    if not is_processing:
        return
    if messagebox.askyesno(
        "確認", "処理を中断しますか？\n(完了していない処理は破棄されます)"
    ):
        print("中断ボタン押下。中断リクエスト送信。")
        stop_requested = True
        if pause_event and not pause_event.is_set():
            pause_event.set()
        pause_resume_button.config(state=tk.DISABLED)
        cancel_button.config(state=tk.DISABLED)
        status_label.config(text="中断処理中...")
        # finalize_processing は update_gui_from_queue または直接のエラー時に呼ばれる


# --- GUIの作成 ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    root.title("サムネイルグリッドジェネレーター")
    root.geometry("750x700")  # 幅を少し広げる
    style = ttk.Style()
    theme_selected = "default"
    try:
        if platform.system() == "Windows":
            theme_selected = "vista"
            style.theme_use(theme_selected)
        else:
            theme_selected = "clam"
            style.theme_use(theme_selected)
    except tk.TclError:
        print("指定テーマが見つからず。デフォルトを使用。")
    status_bg = "#F0F0F0"
    status_fg = "#333333"
    status_font = ("Segoe UI", 10)
    style.configure(
        "StatusBar.TFrame", background=status_bg, relief=tk.FLAT, borderwidth=1
    )
    style.configure(
        "StatusBar.TLabel",
        background=status_bg,
        foreground=status_fg,
        font=status_font,
        padding=(10, 6),
    )
    style.configure(
        "StatusBar.Horizontal.TProgressbar",
        troughcolor="#D3D3D3",
        background="#3477C6",
        thickness=18,
    )
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    input_dir_label = ttk.Label(main_frame, text="入力ディレクトリ:")
    input_dir_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    input_dir_entry = ttk.Entry(main_frame, width=70)
    input_dir_entry.grid(
        row=1, column=0, columnspan=2, padx=5, pady=2, sticky=(tk.W, tk.E)
    )
    input_dir_button = ttk.Button(
        main_frame, text="参照...", command=browse_input_directory
    )
    input_dir_button.grid(row=1, column=2, padx=5, pady=2)
    output_dir_label = ttk.Label(main_frame, text="出力ディレクトリ:")
    output_dir_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    output_dir_entry = ttk.Entry(main_frame, width=70)
    output_dir_entry.grid(
        row=3, column=0, columnspan=2, padx=5, pady=2, sticky=(tk.W, tk.E)
    )
    output_dir_button = ttk.Button(
        main_frame, text="参照...", command=browse_output_directory
    )
    output_dir_button.grid(row=3, column=2, padx=5, pady=2)
    main_frame.grid_columnconfigure(0, weight=1)

    # --- オプションフレーム ---
    options_frame = ttk.LabelFrame(
        main_frame, text="オプション", padding=10
    )  # LabelFrameに変更
    options_frame.grid(
        row=4, column=0, columnspan=3, padx=5, pady=(10, 5), sticky=(tk.W, tk.E)
    )
    options_frame.grid_columnconfigure(1, weight=1)  # 2列目が伸びるように

    # --- グリッド数 ---
    grid_layout_label = ttk.Label(options_frame, text="グリッド数:")
    grid_layout_label.grid(row=0, column=0, padx=(0, 5), pady=3, sticky=tk.W)
    grid_layout_options = [
        "9 (3x3)",
        "16 (4x4)",
        "25 (5x5)",
        "36 (6x6)",
        "49 (7x7)",
        "64 (8x8)",
        "81 (9x9)",
        "100 (10x10)",
    ]
    grid_layout_combobox = ttk.Combobox(
        options_frame, values=grid_layout_options, state="readonly", width=10
    )
    grid_layout_combobox.set("16 (4x4)")
    grid_layout_combobox.grid(row=0, column=1, padx=5, pady=3, sticky=tk.W)

    # --- 出力解像度 ---
    size_label = ttk.Label(options_frame, text="出力解像度:")
    size_label.grid(row=0, column=2, padx=(10, 5), pady=3, sticky=tk.W)
    size_options = ["854x480", "1280x720", "1920x1080", "3840x2160", "7680x4320"]
    size_combobox = ttk.Combobox(
        options_frame, values=size_options, state="readonly", width=12
    )
    size_combobox.set("1920x1080")
    size_combobox.grid(row=0, column=3, padx=5, pady=3, sticky=tk.W)

    # --- プロセス数 ---
    process_count_label = ttk.Label(options_frame, text="使用プロセス数:")
    process_count_label.grid(row=1, column=0, padx=(0, 5), pady=3, sticky=tk.W)
    cpu_count = multiprocessing.cpu_count()
    process_options = ["自動"] + [
        str(i) for i in [1, 2, 4, 8, 12, 16] if i <= cpu_count
    ]  # CPUコア数以下で選択肢生成
    process_count_combobox = ttk.Combobox(
        options_frame, values=process_options, state="readonly", width=10
    )
    process_count_combobox.set("自動")
    process_count_combobox.grid(row=1, column=1, padx=5, pady=3, sticky=tk.W)

    # --- チェックボックス (オプションフレーム内に移動) ---
    checkbox_frame = ttk.Frame(options_frame)  # オプションフレーム内に配置
    checkbox_frame.grid(
        row=1, column=2, columnspan=2, padx=10, pady=3, sticky=tk.W
    )  # column=2から開始
    use_gpu_var = tk.BooleanVar(value=False)
    use_gpu_check = ttk.Checkbutton(
        checkbox_frame, text="GPUアクセラレーション (OpenCL)", variable=use_gpu_var
    )
    use_gpu_check.pack(side=tk.LEFT, padx=5)
    open_explorer_var = tk.BooleanVar(value=True)
    open_explorer_check = ttk.Checkbutton(
        checkbox_frame, text="終了時にフォルダを開く", variable=open_explorer_var
    )
    open_explorer_check.pack(side=tk.LEFT, padx=5)

    # --- 制御ボタン ---
    control_button_frame = ttk.Frame(main_frame)
    control_button_frame.grid(row=7, column=0, columnspan=3, pady=10)  # row変更
    start_button = ttk.Button(
        control_button_frame, text="処理開始", command=start_processing, padding=(10, 5)
    )
    start_button.pack(side=tk.LEFT, padx=10)
    pause_resume_button = ttk.Button(
        control_button_frame,
        text="一時停止",
        command=pause_resume_processing,
        state=tk.DISABLED,
        padding=(10, 5),
    )
    pause_resume_button.pack(side=tk.LEFT, padx=10)
    cancel_button = ttk.Button(
        control_button_frame,
        text="中断",
        command=cancel_processing,
        state=tk.DISABLED,
        padding=(10, 5),
    )
    cancel_button.pack(side=tk.LEFT, padx=10)

    # --- 結果表示エリア ---
    result_frame = ttk.LabelFrame(main_frame, text="処理ログ", padding="5")
    result_frame.grid(
        row=8, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S)
    )  # row変更
    main_frame.grid_rowconfigure(8, weight=1)
    result_text = tk.Text(
        result_frame, width=80, height=15, wrap=tk.NONE, state=tk.DISABLED
    )
    result_scrollbar_y = ttk.Scrollbar(
        result_frame, orient=tk.VERTICAL, command=result_text.yview
    )
    result_scrollbar_x = ttk.Scrollbar(
        result_frame, orient=tk.HORIZONTAL, command=result_text.xview
    )
    result_text.config(
        yscrollcommand=result_scrollbar_y.set, xscrollcommand=result_scrollbar_x.set
    )
    result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    result_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
    result_scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
    result_frame.grid_rowconfigure(0, weight=1)
    result_frame.grid_columnconfigure(0, weight=1)

    # --- ステータスバー ---
    status_frame = ttk.Frame(root, style="StatusBar.TFrame")
    status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0), ipady=3)
    status_label = ttk.Label(
        status_frame, text="準備完了", anchor=tk.W, style="StatusBar.TLabel"
    )
    status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
    progress_bar = ttk.Progressbar(
        status_frame,
        orient="horizontal",
        mode="determinate",
        style="StatusBar.Horizontal.TProgressbar",
    )
    progress_bar.pack(
        side=tk.RIGHT, fill=tk.X, expand=False, padx=(5, 10), pady=8, ipadx=150
    )

    root.mainloop()
