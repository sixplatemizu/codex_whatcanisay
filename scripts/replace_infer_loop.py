# ruff: noqa: E501

p = "src/app/main.py"
with open(p, encoding="utf-8", errors="replace") as _f:
    s = _f.read()
start = s.find("def infer_loop()")
if start == -1:
    print("infer_loop not found")
    raise SystemExit(1)
# Heuristic end: before '\n    # 校准事件处理' or '\n    # \' (layout marker) or 'start_btn.on_click'
end_markers = [
    "\n    # 校准事件处理",
    "\n    start_btn.on_click",
    "\n    # \xe9\x83\xa8\xe5\xb1\x80",
]
end = -1
for m in end_markers:
    pos = s.find(m, start)
    if pos != -1:
        end = pos
        break
if end == -1:
    print("end marker not found")
    raise SystemExit(2)

prefix = s[:start]
suffix = s[end:]

new_body = """def infer_loop() -> None:
        nonlocal state
        tracker: Optional[FaceTracker] = None
        last_ui = 0.0
        last_infer_ms: float = 0.0
        last_encode_ms: float = 0.0
        q = state.frame_queue
        if q is None:
            return
        try:
            while state.capturing and state._stop_event and not state._stop_event.is_set():
                try:
                    frame = q.get(timeout=0.05)
                except Exception:
                    continue
                if mirror_switch.value:
                    frame = cv2.flip(frame, 1)
                if tracker is None:
                    try:
                        tracker = FaceTracker(infer_width=state.infer_width, smooth_iris=True, smooth_alpha=0.6)
                    except Exception as ex:
                        tracker = None
                        track_switch.value = False
                        status_text.value = f"追踪不可用：{ex}（已自动关闭叠加）"
                        page.update()
                if tracker is not None:
                    try:
                        t0 = time.perf_counter()
                        tr = tracker.process(frame)
                        last_infer_ms = (time.perf_counter() - t0) * 1000.0
                        if tr is not None:
                            if track_switch.value:
                                FaceTracker.draw_overlays(frame, tr)
                            try:
                                iris_px = average_iris_center(tr.iris_centers)
                            except Exception:
                                iris_px = None
                            state.last_iris_px = iris_px
                            h0, w0 = frame.shape[:2]
                            if state.cal_targets and state.cal_idx < len(state.cal_targets):
                                tu, tv = state.cal_targets[state.cal_idx]
                                cx, cy = int(tu * w0), int(tv * h0)
                                cv2.circle(frame, (cx, cy), 8, (0, 170, 255), thickness=2, lineType=cv2.LINE_AA)
                            if state.task_running and state.rois:
                                for r in state.rois or []:
                                    x0 = int(r.x0 * w0); y0 = int(r.y0 * h0)
                                    x1 = int(r.x1 * w0); y1 = int(r.y1 * h0)
                                    cv2.rectangle(frame, (x0, y0), (x1, y1), (100, 200, 255), 2, lineType=cv2.LINE_AA)
                            if state.cal_collect_frames_remaining and state.cal_collect_frames_remaining > 0 and iris_px is not None:
                                if state.cal_collect_buffer is None:
                                    state.cal_collect_buffer = []
                                state.cal_collect_buffer.append(iris_px)
                                state.cal_collect_frames_remaining -= 1
                                if state.cal_collect_frames_remaining <= 0 and state.cal_collect_buffer:
                                    xs = [p[0] for p in state.cal_collect_buffer]
                                    ys = [p[1] for p in state.cal_collect_buffer]
                                    avg_src = (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))
                                    if state.cal_targets and state.cal_idx < len(state.cal_targets) and state.calibrator is not None:
                                        dst_uv = state.cal_targets[state.cal_idx]
                                        state.calibrator.add_sample(avg_src, dst_uv)
                                        state.cal_idx += 1
                                        status_text.value = f"已采样 {state.calibrator.num_samples} 个点"
                                        try:
                                            cal_sample_btn.disabled = False if state.cal_idx < len(state.cal_targets) else True
                                            cal_fit_btn.disabled = False if state.calibrator.num_samples >= 3 else True
                                        except Exception:
                                            pass
                                        page.update()
                    except Exception as ex:
                        track_switch.value = False
                        status_text.value = f"推理错误：{ex}（已关闭叠加）"
                        page.update()

                # 收集 gaze 与任务
                now = time.perf_counter()
                try:
                    if state.gaze_series is not None:
                        t_ms = now * 1000.0
                        h0, w0 = frame.shape[:2]
                        if state.last_iris_px is not None:
                            if state.cal_model is not None:
                                u, v = state.cal_model.predict(state.last_iris_px)
                            else:
                                u = float(state.last_iris_px[0]) / float(w0)
                                v = float(state.last_iris_px[1]) / float(h0)
                            state.gaze_series.append((t_ms, u, v, True))
                            state.last_gaze_uv = (u, v)
                            if gaze_dot_switch.value:
                                try:
                                    gx, gy = int(u * w0), int(v * h0)
                                    cv2.circle(frame, (gx, gy), 5, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                                except Exception:
                                    pass
                        else:
                            state.gaze_series.append((t_ms, 0.0, 0.0, False))
                            state.last_gaze_uv = None
                        if len(state.gaze_series) > 5000:
                            state.gaze_series = state.gaze_series[-4000:]
                        if state.task_running and (t_ms >= state.task_end_ms):
                            state.task_running = False
                            try:
                                if state.rois:
                                    result = compute_metrics_for_rois(state.gaze_series, state.rois, min_fix_ms=120.0)
                                    status_text.value = (
                                        f"任务结束 | 左停留 {int(result['left']['dwell_ms'])}ms, 右停留 {int(result['right']['dwell_ms'])}ms; "
                                        f"左首注 {result['left']['first_fix_ms']}ms, 右首注 {result['right']['first_fix_ms']}ms"
                                    )
                                    if state.session_logger is not None:
                                        state.session_logger.set_metrics(task_metrics=result)
                                        state.session_logger.add_event("task_end")
                                        state.session_logger.save()
                            except Exception:
                                pass
                    # UI 更新
                    if now - last_ui >= 1.0 / 30.0:
                        jpg_q = 85 if state.fps < 20.0 else (70 if state.fps > 40.0 else 80)
                        t_enc0 = time.perf_counter()
                        img.src_base64 = frame_to_jpeg_base64(frame, max_width=960, quality=jpg_q)
                        last_encode_ms = (time.perf_counter() - t_enc0) * 1000.0
                        if state.last_gaze_uv is not None and gaze_dot_switch.value:
                            gu, gv = state.last_gaze_uv
                            metrics_text.value = f"FPS: {state.fps:.1f} | I {last_infer_ms:.1f}ms | E {last_encode_ms:.1f}ms | G {gu:.2f},{gv:.2f}"
                        else:
                            metrics_text.value = f"FPS: {state.fps:.1f} | I {last_infer_ms:.1f}ms | E {last_encode_ms:.1f}ms"
                        page.update()
                        last_ui = now
                except Exception:
                    pass
        finally:
            if tracker is not None:
                tracker.close()
"""

new_s = prefix + new_body + suffix
with open(p, "w", encoding="utf-8", newline="") as _f:
    _f.write(new_s)
print("infer_loop replaced")
