# **Ouroboros-Diffusion: Exploring Consistent Content Generation in Tuning-Free Long Video Diffusion**  

🚀 **Official Implementation of Ouroboros-Diffusion (AAAI 2025)**  

This repository provides the implementation of **Ouroboros-Diffusion**, a novel **tuning-free long video diffusion framework** designed to enhance **structural and subject consistency** in long video generation.  

📄 **Our paper has been accepted at AAAI 2025!**  

[📜 Paper (Arxiv)](https://arxiv.org/abs/2501.09019) | [🔗 Project Page (Coming Soon)](TBD) | [🎥 Demo Videos (Coming Soon)](TBD)

---

## **📝 Overview**  
Current **tuning-free video diffusion** methods (e.g., FIFO-Diffusion) struggle with **long-term consistency**, leading to **flickering frames and subject appearance drift**. Ouroboros-Diffusion addresses this by introducing **three key components**:  

✅ **Coherent Tail Latent Sampling** → Ensures **structural continuity** by replacing independent Gaussian noise with a **low-frequency-preserving latent**.  
✅ **Subject-Aware Cross-Frame Attention (SACFA)** → Aligns **subject appearances across frames**, improving **short-term consistency**.  
✅ **Self-Recurrent Guidance** → Utilizes **historical frame embeddings** to **guide denoising**, ensuring **long-term subject coherence**.  

These innovations **significantly improve video consistency**, outperforming **StreamingT2V, FIFO-Diffusion, and FreeNoise** on the **VBench benchmark**.  

📊 **Key Results**  
✔ **Higher subject consistency**  
✔ **Smoother motion transitions**  
✔ **Reduced flickering & content drift**  

---

## **🛠️ Code Release Plan**  

🔹 We will **first release three code demos**, each corresponding to one of the key components:  
1️⃣ **Coherent Tail Latent Sampling**  
2️⃣ **Subject-Aware Cross-Frame Attention (SACFA)**  
3️⃣ **Self-Recurrent Guidance**  

📅 **Full implementation** is expected to be released **before March 15th, 2025**. Stay tuned! 🚀  

---

## **📌 Setup & Installation**  

### **1️⃣ Install Dependencies**  
```bash
git clone https://github.com/your-repo/Ouroboros-Diffusion.git
cd Ouroboros-Diffusion
pip install -r requirements.txt
```

### **2️⃣ Run Demo Codes**  
Each component has its own demo script:  
```bash
python demo_coherent_tail_latent.py  # Run Coherent Tail Latent Sampling
python demo_sacfa.py                 # Run Subject-Aware Cross-Frame Attention
python demo_self_recurrent.py         # Run Self-Recurrent Guidance
```

### **3️⃣ Full Pipeline (Coming Soon)**  
The full Ouroboros-Diffusion pipeline will be available by **March 15th, 2025**.

---

## **📊 Results & Evaluation**  

Ouroboros-Diffusion achieves **state-of-the-art performance** on the **VBench benchmark**, surpassing **FIFO-Diffusion, StreamingT2V, and FreeNoise** in:  
✅ **Content Consistency**  
✅ **Visual Fidelity**  
✅ **Motion Smoothness**  
✅ **Video-Text Alignment**  


---

## **📌 Citation**  

If you find this work useful, please consider citing:  

```bibtex
@inproceedings{chen2025ouroboros,
  title={Ouroboros-Diffusion: Exploring Consistent Content Generation in Tuning-Free Long Video Diffusion},
  author={Chen, Jingyuan and Long, Fuchen and An, Jie and Qiu, Zhaofan and Yao, Ting and Luo, Jiebo and Mei, Tao},
  booktitle={AAAI},
  year={2025}
}
```

---

## **📩 Contact**  

For questions, feel free to open an issue or reach out:  
📧 **Email:** jchen157@u.rochester.edu  
🐦 **Twitter:** TBD  

🔔 **Stay updated!** Star ⭐ this repo and check back for the full code release! 🚀
