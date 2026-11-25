# Báo cáo: M2 — Policy MLP lớn cho robot Unitree G1 đi bộ

## 1. Mục tiêu

Mục tiêu của M2 là train **robot hình người Unitree G1 đi trên mặt phẳng** và đi theo lệnh vận tốc. Policy là một **MLP lớn** (896-448-128), nhận observation hiện tại `o_t` làm input.

## 2. Thiết lập

**Robot và bộ mô phỏng**
- Robot hình người Unitree G1, có 29 khớp được điều khiển.
- MuJoCo Playground (MJX chạy trên JAX), task `G1JoystickFlatTerrain`.
- Vật lý chạy ở 500 Hz, policy chạy ở 50 Hz. Một episode dài 1000 bước (20 giây).

**Môi trường.** Mình giữ **nguyên config mặc định của Playground, không thay đổi gì**:
- Robot phải đi theo một lệnh vận tốc ngẫu nhiên: vx trong [-1, 1], vy trong [-0.5, 0.5], tốc độ xoay trong [-1, 1].
- Có nhiễu cảm biến, và robot bị đẩy ngẫu nhiên mỗi 5–10 giây.
- Domain randomization: ma sát sàn U(0.4, 1.0), khối lượng các khâu ×U(0.9, 1.1), khối lượng thân ±1 kg.
- Thiết lập duy nhất mình chọn là `impl: jax`. Thiết lập này chỉ chọn backend mô phỏng, không thay đổi vật lý hay reward.

**Observation**
- Actor (`state`, 103 giá trị): vận tốc thân, vận tốc góc, hướng trọng lực, lệnh vận tốc, góc khớp, vận tốc khớp, action của bước trước, và pha dáng đi.
- Critic (`privileged_state`, 216 giá trị): toàn bộ input của actor, cộng thêm một số giá trị thật, ví dụ lực tiếp xúc và độ cao thân. Cách làm này gọi là asymmetric actor-critic.

**Huấn luyện (PPO, Brax).** Mình dùng bộ tham số PPO chính thức của Playground cho G1:

| Tham số | Giá trị |
|---|---|
| Tổng số bước | 100M |
| Số env song song | 8192 |
| Unroll length | 20 |
| Batch size × số minibatch | 256 × 32 |
| Số lần update mỗi batch | 4 |
| Learning rate | 3e-4 |
| Discount γ / GAE λ | 0.97 / 0.95 |
| Clip ε | 0.2 |
| Entropy cost | 0.005 |
| Phần cứng | 2× NVIDIA T4 (Kaggle) |

## 3. Kiến trúc M2

| Phần | Input | Các lớp ẩn | Output | Số tham số |
|---|---|---|---|---|
| Actor | `state` (103) | 896-448-128 | 58 (mean + std cho 29 khớp) | 559,994 |
| Critic | `privileged_state` (216) | 512-256-128 | 1 (value) | 275,457 |

- Hàm kích hoạt là swish, phân phối action là tanh-normal.
- Observation được chuẩn hóa bằng thống kê chạy (running statistics). Bộ chuẩn hóa không được tính vào số tham số.

## 4. Những gì đã làm đến giờ

**Các lần chạy thử đầu tiên (config cũ).** Trước khi chốt thiết lập cuối, mình đã chạy hai lần train 50M bước với actor nhỏ hơn (512-256-128):

| Lần chạy | Thay đổi | Kết quả |
|---|---|---|
| 1 | Lệnh mặc định | Trong video, robot có vẻ đi lùi. |
| 2 | Chỉ cho lệnh đi tới (vx trong [0.3, 1.0]) | Độ dài episode khi eval ở cuối khoảng 340/1000 (318/1000 ở một lần chạy lặp lại). Trong video, robot ngã sau 39 bước. |

**Các vấn đề tìm được**
1. **Train chưa đủ.** Độ dài episode vẫn đang tăng ở cuối, nghĩa là policy chưa hội tụ. Ngân sách chính thức cho G1 lớn hơn 50M bước rất nhiều.
2. **Lỗi trong script làm video demo.** Script dùng dải lệnh mặc định chứ không dùng thiết lập lúc train, và cũng không hiện lệnh ra. Vì vậy việc robot "đi lùi" ở lần 1 nhiều khả năng là robot đang đi đúng theo một lệnh đi lùi được chọn ngẫu nhiên. Ở lần 2, robot nhận những lệnh mà nó chưa từng gặp lúc train.
3. **Đổi dải lệnh vận tốc là đổi môi trường.** Mình đã đổi lại như cũ để task vẫn là task chuẩn.

**Cách sửa**
- Môi trường đã trở về config mặc định của Playground.
- Tham số PPO giờ là bộ chính thức, và mạng M2 đã được thiết lập.
- Video demo dùng cùng môi trường như lúc train, với một lệnh cố định (đi tới 0.5 m/s) được in ra trong output.
- Script train giờ ghi lại số tham số, kiểm tra các thiết lập batch, và có thể train tiếp từ checkpoint.

## 5. Tình trạng hiện tại

M2 đang được **train lại từ đầu** với thiết lập cuối cùng (100M bước). Kết quả sẽ được thêm vào đây sau khi train xong.

**Cách đánh giá thành công**
- Độ dài episode khi eval tiến gần tới 1000 (không ngã trong 20 giây).
- Sai số bám vận tốc thấp.
- Robot đi tới mượt mà trong video demo với lệnh 0.5 m/s.

**Các bài test dự kiến sau khi train**
- Sai số bám vận tốc và tỉ lệ ngã.
- Khả năng hồi phục sau khi bị đẩy.
- Đi trên sàn có ma sát khác nhau, kể cả các giá trị nằm ngoài dải đã train.
- Thời gian train và chi phí inference.

## 6. Ghi chú và hạn chế

- Input của actor đã có sẵn **action của bước trước**, nên policy có một chút "trí nhớ" rất ngắn, đúng một bước.
- Một seed là chưa đủ. Dự kiến chạy ít nhất 3 seed.
- Trên Kaggle, phiên bản thư viện có thể thay đổi giữa các phiên, nên mình ghi lại phiên bản cho mỗi lần chạy.
