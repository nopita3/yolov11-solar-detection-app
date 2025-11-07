import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# ตั้งค่า page config
st.set_page_config(
    page_title="Solar Anomaly Detection",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS สำหรับการออกแบบ
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #FF8C00;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .mode-selector {
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# โลโก้และหัวข้อ
st.markdown('<div class="main-header">🌞 Solar Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown("---")

# ตรวจสอบและโหลด model
@st.cache_resource
def load_model():
    """โหลด YOLOv8 model"""
    try:
        from ultralytics import YOLO
        model_path = "best.pt"
        
        if not os.path.exists(model_path):
            st.error(f"❌ ไม่พบไฟล์ model: {model_path}")
            st.info("โปรดตรวจสอบว่า best.pt อยู่ในโฟลเดอร์เดียวกัน")
            return None
        
        model = YOLO(model_path)
        return model
    except ImportError:
        st.error("❌ ไม่พบ ultralytics library")
        st.info("กรุณารัน: pip install ultralytics")
        return None
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลด model: {str(e)}")
        return None

# ฟังก์ชันสำหรับการทำนาย
def predict_anomaly(image, model):
    """ทำนายผลใช้ model"""
    if model is None:
        st.error("❌ Model ยังไม่ได้โหลด")
        return None
    # แปลง PIL Image เป็น numpy array ถ้าจำเป็น
    if isinstance(image, Image.Image):
        image = np.array(image)

    # ทำนาย
    results = model(image)
    r = results[0]

    # ถ้าเป็น classifier (ไม่มี boxes) ให้วาด probs เองบนภาพ
    if (r.boxes is None or len(r.boxes) == 0) and getattr(r, "probs", None) is not None:
        img = getattr(r, "orig_img", image)
        probs = np.array(r.probs.data.cpu().numpy()).ravel()
        names = getattr(r, "names", {}) or {}

        idxs = sorted(range(len(probs)), key=lambda i: float(probs[i]), reverse=True)
        lines = [f"{(names.get(i) if isinstance(names, dict) else names[i])} {float(probs[i]):.2f}" for i in idxs]

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        y0, dy = 30, 28
        for j, text in enumerate(lines):
            y = y0 + j * dy
            cv2.putText(img_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        annotated_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return annotated_frame, r

    # ปกติใช้ plot()
    annotated_frame = r.plot()
    return annotated_frame, r

# ฟังก์ชันสำหรับแสดงผลลัพธ์
def display_results(annotated_frame, results):
    """แสดงผลลัพธ์การทำนาย"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(annotated_frame, caption="ผลการทำนาย", width='stretch')
    
    with col2:
        st.subheader("📊 รายละเอียด")
        if results.boxes is not None and len(results.boxes) > 0:
            st.success(f"✅ พบความผิดปกติ: {len(results.boxes)} ตัว")
            for idx, box in enumerate(results.boxes):
                st.write(f"**ความผิดปกติที่ {idx + 1}:**")
                confidence = float(box.conf[0])
                st.write(f"- ความมั่นใจ: {confidence:.2%}")
                class_id = int(box.cls[0])
                st.write(f"- ประเภท: Class {class_id}")
        else:
            # classifier-style results: read probs directly and display
            probs = np.array(results.probs.data.cpu().numpy()).ravel()
            names = getattr(results, "names", {}) or {}

            rows = []
            for i, p in enumerate(probs):
                label = names.get(i, str(i)) if isinstance(names, dict) else names[i]
                score = float(p)
                rows.append({"class": label, "probability": f"{score:.2%}", "score": score})

            # top-1
            top_idx = int(np.argmax(probs))
            st.success(f"🔎 ทำนาย: {names.get(top_idx, str(top_idx))} ({probs[top_idx]:.2%})")

            # show sorted list (match overlay) and table
            sorted_idx = sorted(range(len(rows)), key=lambda i: rows[i]["score"], reverse=True)
            st.markdown("**🔢 ค่า probability (เรียงจากมาก->น้อย):**")
            for i in sorted_idx:
                st.write(f"{rows[i]['class']} {rows[i]['score']:.2f}")

            st.subheader("📝 ความน่าจะเป็นของแต่ละคลาส")
            st.table(rows)

    
# Sidebar - เลือกโหมด
st.sidebar.markdown("## ⚙️ ตั้งค่า")
mode = st.sidebar.radio(
    "เลือกโหมดการใช้งาน:",
    ["📁 อัปโหลดรูป", "📹 อัปโหลดวิดีโอ"],
    key="mode_selector"
)

# โหลด model
model = load_model()

# ============================================
# โหมด 1: อัปโหลดรูป
# ============================================
if mode == "📁 อัปโหลดรูป":
    st.header("📁 อัปโหลดรูปภาพ")
    
    uploaded_image = st.file_uploader(
        "เลือกรูปภาพ:",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader"
    )
    
    if uploaded_image is not None:
        # แสดงรูปต้นฉบับ
        image = Image.open(uploaded_image)
        st.image(image, caption="รูปต้นฉบับ", width='stretch')
        
        # ปุ่มการทำนาย
        if st.button("🔍 ทำนายผล", key="predict_button"):
            with st.spinner("⏳ กำลังวิเคราะห์..."):
                result = predict_anomaly(image, model)
                
                if result and result[0] is not None:
                    annotated_frame, results = result
                    display_results(annotated_frame, results)
                    
                    # บันทึกรูปลัพธ์
                    if st.button("💾 บันทึกผลลัพธ์"):
                        output_image = Image.fromarray(annotated_frame)
                        output_image.save("result_anomaly_detection.png")
                        st.success("✅ บันทึกรูปลัพธ์เสร็จสิ้น")

# ============================================
# โหมด 2: อัปโหลดวิดีโอ
# ============================================
elif mode == "📹 อัปโหลดวิดีโอ":
    st.header("📹 อัปโหลดวิดีโอ")
    
    uploaded_video = st.file_uploader(
        "เลือกไฟล์วิดีโอ:",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        # บันทึกวิดีโอชั่วคราว
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_video.read())
        temp_file.close()
        
        # แสดงวิดีโอต้นฉบับ
        st.video(temp_file.name)
        
        # ปุ่มการวิเคราะห์
        if st.button("🔍 วิเคราะห์วิดีโอ", key="analyze_video_button"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            result_container = st.container()
            
            # เปิดวิดีโอ
            cap = cv2.VideoCapture(temp_file.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            detected_anomalies = 0
            processed_frames = []
            
            with st.spinner(f"⏳ กำลังประมวลผล {total_frames} เฟรม..."):
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # ประมวลผลทุก 5 เฟรม เพื่อความเร็ว
                    if frame_count % 5 == 0:
                        # แปลงสีจาก BGR เป็น RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # ทำนาย
                        result = predict_anomaly(frame_rgb, model)
                        
                        if result and result[0] is not None:
                            annotated_frame, results = result
                            
                            if results.boxes is not None and len(results.boxes) > 0:
                                detected_anomalies += len(results.boxes)
                            
                            if len(processed_frames) < 5:  # เก็บ 5 เฟรม
                                processed_frames.append(annotated_frame)
                    
                    # อัปเดตแถบความคืบหน้า
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"📊 ประมวลผล: {frame_count}/{total_frames} เฟรม ({progress:.1%})")
            
            cap.release()
            
            # แสดงผลลัพธ์
            with result_container:
                st.subheader("📊 ผลการวิเคราะห์วิดีโอ")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📹 จำนวนเฟรมทั้งหมด", total_frames)
                with col2:
                    st.metric("⏱️ ระยะเวลา (วินาที)", f"{total_frames/fps:.1f}")
                with col3:
                    st.metric("🚨 ความผิดปกติที่พบ", detected_anomalies)
                
                if len(processed_frames) > 0:
                    st.subheader("📸 ตัวอย่างเฟรมที่ประมวลผล")
                    cols = st.columns(min(3, len(processed_frames)))
                    for idx, frame in enumerate(processed_frames):
                        with cols[idx % len(cols)]:
                            st.image(frame, caption=f"เฟรมที่ {idx*5}")
        
        # ลบไฟล์ชั่วคราว
        try:
            os.remove(temp_file.name)
        except Exception:
            pass

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    🌞 Solar Anomaly Detection System | Powered by YOLOv11n-cls
    </div>
""", unsafe_allow_html=True)
