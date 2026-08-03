"""Generate 2,000 deterministic, unreviewed Vietnamese pilot SFT drafts.

This script uses only project-local templates and authored fact tables. It never
marks a record approved; a human must replace ``reviewer-unassigned`` and set
``review_status`` to ``approved`` after reviewing each record.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from viemmo.data.contamination import find_contamination, find_duplicates  # noqa: E402
from viemmo.data.pii import record_pii  # noqa: E402
from viemmo.data.pipeline import (  # noqa: E402
    chat_token_count,
    load_jsonl,
    load_local_tokenizer,
    write_jsonl,
)
from viemmo.data.schema import (  # noqa: E402
    SCHEMA_VERSION,
    SOURCE,
    SYSTEM_PROMPT,
    validate_records,
)


VARIANTS = (
    ("nhóm biên tập", "buổi họp sáng", "bản thảo hướng dẫn", "đầu tuần"),
    ("tổ kỹ thuật", "đợt kiểm thử nội bộ", "báo cáo thử nghiệm", "cuối tuần"),
    ("câu lạc bộ đọc sách", "sinh hoạt tháng", "danh mục tài liệu", "chiều thứ Sáu"),
    ("nhóm tình nguyện", "hoạt động cộng đồng", "kế hoạch phân công", "sáng Chủ nhật"),
    ("ban tổ chức", "hội thảo nhỏ", "thông báo tham dự", "giữa tháng"),
    ("tổ thư viện", "đợt sắp xếp kho", "phiếu kiểm kê", "sau giờ nghỉ"),
    ("nhóm thiết kế", "lần rà soát giao diện", "bản mẫu mới", "chiều mai"),
    ("đội vận hành", "ca bảo trì", "danh sách kiểm tra", "trước giờ mở cửa"),
    ("lớp học", "buổi thực hành", "tài liệu bài tập", "tiết cuối"),
    ("nhóm nghiên cứu", "phiên trao đổi", "bản ghi kết quả", "cuối tháng"),
)


GRAMMAR = (
    ("dấu phẩy sau trạng ngữ", "Sau khi rà soát tài liệu, {actor} gửi {obj}.", "Dấu phẩy tách trạng ngữ đầu câu khỏi nòng cốt câu."),
    ("dấu hai chấm trước phần liệt kê", "{actor} cần ba thứ: lịch làm việc, {obj} và sổ ghi chép.", "Dấu hai chấm báo hiệu phần liệt kê theo sau."),
    ("dấu chấm phẩy giữa hai vế liên quan", "{actor} đã hoàn tất {obj}; phần minh họa vẫn cần chỉnh sửa.", "Dấu chấm phẩy tách hai vế độc lập nhưng có quan hệ chặt chẽ."),
    ("cặp quan hệ từ nguyên nhân–kết quả", "Vì {context} kéo dài nên {actor} điều chỉnh {obj}.", "Cặp “vì ... nên ...” nối nguyên nhân với kết quả."),
    ("cặp quan hệ từ điều kiện–kết quả", "Nếu {actor} hoàn thành {obj} đúng hạn thì buổi tổng kết sẽ diễn ra thuận lợi.", "Cặp “nếu ... thì ...” thể hiện điều kiện và hệ quả."),
    ("từ chỉ thời đã–đang–sẽ", "{actor} đã đọc yêu cầu, đang sửa {obj} và sẽ gửi lại vào {time}.", "Ba từ lần lượt định vị hành động trước, trong và sau thời điểm nói."),
    ("phạm vi của từ phủ định", "{actor} không chỉ sửa {obj} mà còn giải thích lý do thay đổi.", "Cấu trúc này phủ định giới hạn “chỉ”, không phủ định hành động sửa."),
    ("cấu trúc song hành", "{actor} vừa kiểm tra {obj}, vừa ghi lại các điểm cần cải thiện.", "Hai vế có cấu trúc cân xứng và cùng phụ thuộc vào chủ ngữ."),
    ("đại từ quy chiếu rõ ràng", "Sau khi nhận {obj}, trưởng nhóm phản hồi rằng tài liệu này cần thêm ví dụ.", "Cụm “tài liệu này” chỉ rõ đối tượng, tránh đại từ mơ hồ."),
    ("trật tự từ tự nhiên", "Vào {time}, {actor} sẽ trình bày {obj} trong {context}.", "Trạng ngữ thời gian, chủ ngữ, vị ngữ và hoàn cảnh được sắp xếp rõ ràng."),
    ("lời dẫn trực tiếp", "Trưởng nhóm nói: “Hãy gửi {obj} vào {time}.”", "Dấu hai chấm và ngoặc kép đánh dấu lời dẫn trực tiếp."),
    ("lời dẫn gián tiếp", "Trưởng nhóm đề nghị {actor} gửi {obj} vào {time}.", "Lời nói được thuật lại mà không dùng ngoặc kép."),
    ("cách dùng từ “mỗi”", "Mỗi thành viên của {actor} phụ trách một phần trong {obj}.", "“Mỗi” nhấn mạnh từng cá thể trong một tập hợp."),
    ("cách dùng từ “những”", "Những góp ý từ {context} đã giúp {actor} hoàn thiện {obj}.", "“Những” biểu thị một tập hợp sự vật đã được xác định theo ngữ cảnh."),
    ("so sánh hơn", "Bản {obj} mới rõ ràng hơn phiên bản trước.", "“Hơn” tạo quan hệ so sánh giữa hai đối tượng."),
    ("so sánh ngang bằng", "Phần hướng dẫn trong {obj} dễ hiểu như ví dụ trình bày tại {context}.", "“Như” thể hiện mức độ tương đồng giữa hai đối tượng."),
    ("câu chủ động", "{actor} hoàn thiện {obj} vào {time}.", "Chủ ngữ trực tiếp thực hiện hành động."),
    ("câu bị động", "{obj} được {actor} hoàn thiện vào {time}.", "Đối tượng chịu tác động được đưa lên làm chủ ngữ ngữ pháp."),
    ("phép nối bổ sung", "{actor} đã sửa nội dung; ngoài ra, nhóm còn bổ sung hình minh họa cho {obj}.", "“Ngoài ra” liên kết câu sau với một ý bổ sung."),
    ("phép nối tương phản", "{obj} khá đầy đủ; tuy nhiên, {actor} vẫn cần kiểm tra lại nguồn dẫn.", "“Tuy nhiên” báo hiệu ý tương phản với nhận xét trước."),
    ("lược bỏ từ thừa", "{actor} cùng nhau hoàn thiện {obj}.", "Câu có thể gọn hơn thành “{actor} hoàn thiện {obj}” vì “nhóm” đã hàm ý nhiều người cùng làm."),
    ("văn phong trang trọng", "Kính đề nghị {actor} xác nhận thời gian nhận {obj}.", "Cách diễn đạt dùng từ lịch sự, phù hợp trao đổi công việc."),
    ("văn phong thân mật", "Mọi người xem giúp {obj} rồi góp ý nhé.", "Từ “nhé” và cách xưng hô tạo sắc thái gần gũi."),
    ("câu cầu khiến lịch sự", "Vui lòng gửi {obj} cho {actor} trước {time}.", "“Vui lòng” làm yêu cầu trực tiếp trở nên lịch sự."),
    ("câu hỏi lựa chọn", "{actor} muốn trình bày {obj} trong {context} hay gửi trước để mọi người đọc?", "Từ “hay” nêu hai phương án để người nghe lựa chọn."),
    ("thành phần chú thích", "{actor}, đơn vị phụ trách {context}, sẽ cập nhật {obj}.", "Cụm giữa hai dấu phẩy bổ sung thông tin cho chủ ngữ."),
    ("tiêu đề ngắn gọn", "Cập nhật {obj} cho {context}", "Tiêu đề nêu hành động và đối tượng chính, không dùng từ rườm rà."),
    ("trình tự trước–sau", "Trước hết, {actor} kiểm tra {obj}; sau đó, nhóm thảo luận tại {context}.", "Các từ nối làm rõ thứ tự của hai hành động."),
    ("cặp từ không những–mà còn", "{obj} không những dễ đọc mà còn thuận tiện cho {actor} tra cứu.", "Cặp từ nối hai đặc điểm bổ sung và nhấn mạnh đặc điểm sau."),
    ("câu ghép nguyên nhân", "{actor} lùi thời hạn vì {context} cần thêm thời gian chuẩn bị {obj}.", "Quan hệ nguyên nhân được thể hiện bằng từ “vì”."),
)


TECH = (
    ("bộ nhớ đệm", "lưu tạm dữ liệu thường dùng để giảm độ trễ", "dữ liệu cũ có thể còn tồn tại đến khi hết hạn", "đặt thời hạn phù hợp và có cơ chế làm mới"),
    ("hàng đợi thông điệp", "tách bên gửi khỏi bên xử lý bằng một vùng đệm thông điệp", "thông điệp có thể được giao lại khi xác nhận thất bại", "thiết kế tác vụ có tính lặp an toàn"),
    ("giao dịch cơ sở dữ liệu", "gom nhiều thay đổi thành một đơn vị xác nhận hoặc hoàn tác", "giao dịch dài có thể giữ khóa và làm giảm thông lượng", "giữ giao dịch ngắn và xử lý lỗi rõ ràng"),
    ("chỉ mục cơ sở dữ liệu", "tạo cấu trúc hỗ trợ tìm bản ghi nhanh hơn", "chỉ mục tốn dung lượng và làm thao tác ghi nặng hơn", "đo kế hoạch truy vấn trước và sau khi thêm chỉ mục"),
    ("khóa chính", "định danh duy nhất mỗi hàng trong một bảng", "giá trị không ổn định gây khó cho quan hệ tham chiếu", "chọn giá trị duy nhất và ít thay đổi"),
    ("khóa ngoại", "ràng buộc giá trị tham chiếu tới một hàng hợp lệ ở bảng khác", "xóa dữ liệu cha thiếu quy tắc có thể thất bại", "quy định rõ hành vi cập nhật và xóa"),
    ("chuẩn hóa Unicode NFC", "đưa các chuỗi tương đương về một dạng mã hóa thống nhất", "NFC không sửa lỗi chính tả hay thay đổi cách đặt dấu theo phong cách", "chuẩn hóa trước khi so sánh và băm văn bản"),
    ("mã hóa UTF-8", "biểu diễn điểm mã Unicode bằng chuỗi byte có độ dài biến đổi", "đọc byte bằng sai bảng mã tạo ra ký tự lỗi", "khai báo và kiểm tra mã hóa ở mọi biên nhập xuất"),
    ("hàm băm mật mã", "ánh xạ dữ liệu thành giá trị đại diện có độ dài cố định", "hàm băm không phải mã hóa có thể giải ngược", "dùng thuật toán hiện đại và muối khi lưu mật khẩu"),
    ("mã hóa đối xứng", "dùng cùng một khóa bí mật để mã hóa và giải mã", "lộ khóa làm mất tính bí mật của dữ liệu", "quản lý khóa tách biệt khỏi dữ liệu"),
    ("chữ ký số", "cho phép kiểm tra nguồn ký và tính toàn vẹn bằng mật mã bất đối xứng", "chữ ký không tự giữ bí mật nội dung", "xác thực khóa công khai trước khi tin kết quả"),
    ("TLS", "bảo vệ dữ liệu trên đường truyền và xác thực máy chủ bằng chứng thư", "bỏ qua lỗi chứng thư làm suy yếu bảo vệ", "bật kiểm tra chứng thư và phiên bản giao thức hiện đại"),
    ("DNS", "ánh xạ tên miền sang thông tin như địa chỉ mạng", "bộ đệm khiến thay đổi không xuất hiện đồng thời", "điều chỉnh TTL trước thay đổi có kế hoạch"),
    ("HTTP GET", "yêu cầu lấy biểu diễn của một tài nguyên", "GET không nên tạo thay đổi quan trọng ở máy chủ", "giữ thao tác an toàn và có thể lưu đệm khi phù hợp"),
    ("HTTP POST", "gửi dữ liệu để máy chủ xử lý hoặc tạo tài nguyên phụ", "gửi lại có thể tạo thao tác trùng", "dùng khóa chống lặp cho tác vụ cần tính duy nhất"),
    ("mã trạng thái HTTP", "mô tả kết quả xử lý yêu cầu theo nhóm chuẩn", "trả mã thành công cho lỗi làm máy khách xử lý sai", "chọn mã phản ánh đúng kết quả và kèm thông báo có cấu trúc"),
    ("REST", "tổ chức giao tiếp quanh tài nguyên và giao diện thống nhất", "gắn nhãn REST không tự bảo đảm API dễ dùng", "thiết kế tài nguyên, phương thức và lỗi nhất quán"),
    ("JSON", "biểu diễn dữ liệu bằng đối tượng, mảng và các giá trị cơ bản", "JSON chuẩn không hỗ trợ chú thích", "xác thực cấu trúc và kiểu dữ liệu ở đầu vào"),
    ("YAML", "biểu diễn dữ liệu phân cấp theo cú pháp dễ đọc", "thụt lề sai có thể đổi cấu trúc", "dùng trình phân tích an toàn và kiểm tra lược đồ"),
    ("biến môi trường", "truyền cấu hình từ môi trường chạy vào chương trình", "bí mật trong biến vẫn có thể lộ qua nhật ký hoặc tiến trình", "giới hạn quyền đọc và không ghi bí mật ra log"),
    ("container", "đóng gói tiến trình cùng phụ thuộc trong môi trường cô lập ở mức hệ điều hành", "container không phải máy ảo và vẫn dùng nhân máy chủ", "dùng ảnh tối thiểu, không chạy quyền cao khi không cần"),
    ("máy ảo", "mô phỏng phần cứng để chạy một hệ điều hành khách", "máy ảo thường tốn tài nguyên hơn container", "chọn khi cần ranh giới hệ điều hành mạnh hơn"),
    ("Git merge", "kết hợp hai lịch sử phát triển và có thể tạo commit hợp nhất", "xung đột vẫn cần con người giải quyết", "kiểm thử kết quả sau khi hợp nhất"),
    ("Git rebase", "phát lại commit trên một nền lịch sử mới", "rebase thay đổi mã định danh commit", "tránh viết lại lịch sử đã chia sẻ nếu chưa phối hợp"),
    ("kiểm thử đơn vị", "kiểm tra một đơn vị logic nhỏ trong môi trường kiểm soát", "nhiều kiểm thử đơn vị không thay thế kiểm thử tích hợp", "giữ ca kiểm thử độc lập và dễ lặp lại"),
    ("kiểm thử tích hợp", "kiểm tra sự phối hợp giữa nhiều thành phần", "lỗi có thể khó khoanh vùng hơn kiểm thử đơn vị", "dùng dữ liệu kiểm thử ổn định và dọn trạng thái sau chạy"),
    ("tích hợp liên tục", "tự động xây dựng và kiểm tra thay đổi thường xuyên", "pipeline xanh không chứng minh phần mềm không còn lỗi", "giữ kiểm tra nhanh, đáng tin và bắt buộc trước hợp nhất"),
    ("ghi nhật ký", "ghi sự kiện giúp quan sát và chẩn đoán hệ thống", "log có thể làm lộ bí mật hoặc dữ liệu cá nhân", "lọc dữ liệu nhạy cảm và đặt thời hạn lưu giữ"),
    ("số liệu giám sát", "đo đại lượng hệ thống theo thời gian để phát hiện xu hướng", "quá nhiều số liệu không có mục tiêu gây nhiễu", "gắn số liệu với mục tiêu dịch vụ và cảnh báo hữu ích"),
    ("sao lưu", "tạo bản sao dữ liệu để phục hồi sau mất mát", "bản sao chưa thử khôi phục có thể không dùng được", "kiểm thử phục hồi định kỳ và giữ bản sao tách biệt"),
    ("RAID", "kết hợp nhiều ổ đĩa để cải thiện khả dụng hoặc hiệu năng tùy cấp", "RAID không thay thế sao lưu", "chọn cấp phù hợp và vẫn duy trì bản sao độc lập"),
    ("phân quyền tối thiểu", "chỉ cấp quyền cần thiết cho nhiệm vụ", "quyền tích lũy theo thời gian làm tăng rủi ro", "rà soát và thu hồi quyền định kỳ"),
    ("xác thực đa yếu tố", "yêu cầu bằng chứng từ nhiều nhóm yếu tố", "mã một lần vẫn có thể bị lừa nhập vào trang giả", "ưu tiên phương thức chống lừa đảo khi có thể"),
    ("quản lý mật khẩu", "tạo và lưu thông tin đăng nhập duy nhất trong kho bảo vệ", "dùng lại mật khẩu làm một vụ lộ ảnh hưởng nhiều tài khoản", "dùng mật khẩu dài, duy nhất và trình quản lý uy tín"),
    ("SQL injection", "xảy ra khi dữ liệu không tin cậy bị ghép thành câu lệnh SQL", "lọc ký tự đơn lẻ không phải biện pháp đầy đủ", "dùng truy vấn tham số hóa và quyền cơ sở dữ liệu tối thiểu"),
    ("XSS", "cho phép nội dung không tin cậy chạy trong ngữ cảnh trang web", "chỉ kiểm tra đầu vào không bảo vệ mọi ngữ cảnh đầu ra", "mã hóa theo ngữ cảnh và áp dụng chính sách nội dung"),
    ("CORS", "cho trình duyệt biết nguồn nào được phép đọc phản hồi xuyên nguồn", "CORS không phải cơ chế xác thực", "chỉ cho phép nguồn, phương thức và header cần thiết"),
    ("độ phức tạp thuật toán", "mô tả tốc độ tăng tài nguyên theo kích thước đầu vào", "hằng số và dữ liệu thực tế vẫn ảnh hưởng hiệu năng", "đo trên tải đại diện bên cạnh phân tích tiệm cận"),
    ("đệ quy", "giải bài toán bằng lời gọi tới trường hợp nhỏ hơn của chính nó", "thiếu điểm dừng gây lặp vô hạn hoặc tràn ngăn xếp", "xác định trường hợp cơ sở và mức sâu tối đa"),
    ("xử lý đồng thời", "cho nhiều tác vụ tiến triển trong cùng khoảng thời gian", "truy cập trạng thái chung không đồng bộ gây điều kiện tranh chấp", "giảm trạng thái dùng chung và dùng cơ chế đồng bộ phù hợp"),
)


SUMMARY_TOPICS = (
    "lịch mở cửa thư viện", "buổi tập huấn", "đợt bảo trì hệ thống", "kế hoạch trồng cây", "lịch giao tài liệu",
    "chương trình đọc sách", "đợt kiểm kê", "buổi thử nghiệm", "kế hoạch triển lãm", "lịch sinh hoạt câu lạc bộ",
    "đợt cập nhật phần mềm", "chuyến tham quan", "buổi họp phụ huynh", "kế hoạch quyên góp", "lịch luyện tập",
    "đợt khảo sát", "buổi giới thiệu sản phẩm", "kế hoạch chuyển phòng", "lịch phát hành bản tin", "đợt vệ sinh thiết bị",
    "buổi thảo luận", "kế hoạch mượn dụng cụ", "lịch nộp báo cáo", "đợt phân loại tài liệu", "buổi hướng dẫn an toàn",
    "kế hoạch ghi hình", "lịch trực hỗ trợ", "đợt thử âm thanh", "buổi tổng kết", "kế hoạch in ấn",
)

SUMMARY_DETAILS = (
    "quầy mượn trả và khu tự học", "tài khoản thực hành và tài liệu hướng dẫn", "máy chủ dự phòng và cửa sổ gián đoạn", "dụng cụ làm vườn và khu đất chuẩn bị", "khâu đóng gói và điểm tiếp nhận",
    "danh sách tác phẩm và người dẫn chương trình", "nhãn phân loại và số liệu kho", "thiết bị cũ và tiêu chí đo hiệu năng", "không gian trưng bày và bảng thuyết minh", "phòng sinh hoạt và danh sách thành viên",
    "gói cài đặt và quy trình quay lui", "phương tiện di chuyển và điểm đón", "phòng họp và tài liệu trao đổi", "vật phẩm tiếp nhận và biên bản bàn giao", "sân tập và dụng cụ bảo hộ",
    "bảng hỏi và phương án lấy mẫu", "khu trình diễn và tài liệu giới thiệu", "sơ đồ chỗ ngồi và thùng lưu trữ", "bài viết và khâu hiệu đính", "bộ lọc và lịch kiểm tra",
    "câu hỏi thảo luận và người điều phối", "phiếu mượn và tình trạng dụng cụ", "biểu mẫu xác nhận và người tổng hợp", "mã phân loại và danh mục tra cứu", "lối thoát hiểm và thiết bị bảo vệ",
    "bối cảnh quay và quyền sử dụng hình ảnh", "ca trực và kênh tiếp nhận yêu cầu", "micro và hệ thống loa", "số liệu kết quả và phần ghi nhận đóng góp", "bản in thử và yêu cầu định dạng",
)


UNCERTAINTY = (
    ("thời tiết có thuận lợi", "địa điểm, ngày cụ thể và dự báo cập nhật", "nguồn khí tượng địa phương"),
    ("thiết bị có còn bảo hành", "mẫu máy, ngày mua và điều khoản bảo hành", "hóa đơn và chính sách của nhà sản xuất"),
    ("một loại thuốc có dùng chung được", "tên thuốc, liều dùng, bệnh nền và đơn hiện tại", "bác sĩ hoặc dược sĩ"),
    ("món ăn có phù hợp với dị ứng", "thành phần, quy trình chế biến và loại dị ứng", "nhà cung cấp và chuyên gia y tế"),
    ("chuyến xe có đến đúng giờ", "tuyến, ngày đi và trạng thái vận hành", "kênh thông báo chính thức của đơn vị vận tải"),
    ("giá sản phẩm có chính xác", "mẫu hàng, thị trường, tình trạng và thời điểm", "bảng giá cập nhật của bên bán"),
    ("một câu trích dẫn thuộc về tác giả nào", "nguyên văn, ngôn ngữ và nguồn xuất hiện", "ấn bản hoặc thư mục trích dẫn đáng tin cậy"),
    ("tệp tải xuống có an toàn", "nguồn tệp, chữ ký, loại tệp và kết quả quét", "nhà phát hành cùng công cụ bảo mật"),
    ("hình ảnh có bị chỉnh sửa", "tệp gốc, siêu dữ liệu và chuỗi lưu giữ", "chuyên gia pháp chứng số"),
    ("đường đi nào nhanh nhất", "điểm đầu, điểm cuối, thời gian và tình trạng giao thông", "bản đồ có dữ liệu trực tiếp"),
    ("một khóa học có được công nhận", "đơn vị cấp, chương trình và mục đích sử dụng", "cơ quan công nhận liên quan"),
    ("máy tính có chạy được phần mềm", "cấu hình máy, phiên bản hệ điều hành và yêu cầu phần mềm", "tài liệu tương thích chính thức"),
    ("dữ liệu có được sao lưu đầy đủ", "phạm vi, thời điểm, nhật ký và kết quả thử phục hồi", "quản trị viên cùng báo cáo sao lưu"),
    ("một tài khoản có bị xâm nhập", "nhật ký đăng nhập, cảnh báo và hoạt động bất thường", "bộ phận an toàn thông tin"),
    ("kết quả khảo sát có đại diện", "cách lấy mẫu, cỡ mẫu và tỷ lệ phản hồi", "báo cáo phương pháp nghiên cứu"),
    ("tin tức đang lan truyền có thật", "nguồn ban đầu, thời điểm và bằng chứng độc lập", "nhiều nguồn tin đáng tin cậy"),
    ("một văn bản còn hiệu lực", "số hiệu, phạm vi áp dụng và ngày kiểm tra", "cơ sở dữ liệu pháp luật chính thức"),
    ("cây trồng đang thiếu chất gì", "loài cây, đất, triệu chứng và lịch chăm sóc", "chuyên gia nông nghiệp hoặc xét nghiệm đất"),
    ("pin thiết bị có cần thay", "chu kỳ sạc, dung lượng đo được và triệu chứng", "công cụ chẩn đoán của nhà sản xuất"),
    ("âm thanh trong bản ghi là của ai", "bản gốc, mẫu đối chiếu hợp lệ và điều kiện thu", "chuyên gia cùng quy trình xác minh danh tính"),
    ("một con số thống kê có đúng", "định nghĩa chỉ số, thời kỳ và nguồn dữ liệu", "báo cáo gốc và phương pháp tính"),
    ("lịch sự kiện có thay đổi", "tên sự kiện, địa điểm và ngày dự kiến", "trang thông báo của ban tổ chức"),
    ("một linh kiện có tương thích", "mã linh kiện, bo mạch và phiên bản phần cứng", "tài liệu tương thích của nhà sản xuất"),
    ("nước có an toàn để uống", "nguồn nước, khu vực và kết quả xét nghiệm gần nhất", "cơ quan cấp nước hoặc y tế địa phương"),
    ("công thức có cho đúng khẩu phần", "số người, kích thước phần và nguyên liệu", "định lượng thử nghiệm thực tế"),
    ("một bản dịch có sát nghĩa", "văn bản gốc, ngữ cảnh và đối tượng độc giả", "người dịch thành thạo chuyên ngành"),
    ("máy chủ có đang quá tải", "CPU, bộ nhớ, độ trễ và tải theo thời gian", "bảng giám sát vận hành"),
    ("một email có phải lừa đảo", "header, tên miền, liên kết và ngữ cảnh gửi", "bộ phận an toàn qua kênh độc lập"),
    ("tệp dữ liệu có đầy đủ", "lược đồ mong đợi, số bản ghi và quy tắc chất lượng", "báo cáo kiểm tra với nguồn gốc dữ liệu"),
    ("kết luận thí nghiệm có lặp lại được", "quy trình, dữ liệu, môi trường và số lần thử", "thí nghiệm tái lập độc lập"),
)


PRIVACY = (
    ("chia sẻ mật khẩu qua tin nhắn", "lộ thông tin đăng nhập và mất khả năng quy trách nhiệm", "cấp tài khoản riêng và dùng kho bí mật được phê duyệt"),
    ("đưa danh sách khách hàng vào công cụ công cộng", "tiết lộ dữ liệu liên hệ cho bên không được phép", "loại định danh và dùng công cụ đã được phê duyệt"),
    ("gửi toàn bộ nhật ký lỗi", "log có thể chứa mã truy cập và nội dung người dùng", "lọc dữ liệu nhạy cảm rồi chỉ gửi phần cần chẩn đoán"),
    ("dùng lại mật khẩu", "một vụ lộ có thể ảnh hưởng nhiều tài khoản", "dùng mật khẩu duy nhất trong trình quản lý mật khẩu"),
    ("mở liên kết đăng nhập trong thư bất ngờ", "trang giả có thể đánh cắp thông tin", "truy cập dịch vụ bằng địa chỉ đã lưu và báo cáo thư"),
    ("cấp quyền quản trị cho tác vụ thường ngày", "quyền quá mức làm tăng hậu quả khi có lỗi", "cấp quyền tối thiểu và nâng quyền có kiểm soát"),
    ("lưu khóa API trong mã nguồn", "khóa có thể bị sao chép qua kho mã và lịch sử", "dùng hệ thống quản lý bí mật và luân chuyển khóa đã lộ"),
    ("ghi dữ liệu cá nhân vào log", "nhiều người hoặc hệ thống có thể đọc log", "ghi định danh giả và áp dụng thời hạn lưu giữ"),
    ("sao chép dữ liệu thật sang môi trường thử", "môi trường thử thường có kiểm soát yếu hơn", "dùng dữ liệu tổng hợp hoặc đã khử định danh"),
    ("chia sẻ tệp bằng liên kết công khai", "người ngoài phạm vi có thể truy cập", "giới hạn người nhận, thời hạn và quyền tải xuống"),
    ("thu thập nhiều trường thông tin để phòng khi cần", "tăng rủi ro và vượt mục đích xử lý", "chỉ thu thập dữ liệu tối thiểu cho mục đích đã nêu"),
    ("giữ dữ liệu vô thời hạn", "rủi ro tích lũy dù mục đích đã kết thúc", "đặt lịch xóa hoặc ẩn danh theo chính sách"),
    ("gửi nhầm tệp đính kèm", "người nhận thấy dữ liệu không dành cho họ", "kiểm tra người nhận, phân loại và dùng cơ chế thu hồi"),
    ("đọc dữ liệu của đồng nghiệp vì tò mò", "truy cập không có mục đích công việc", "chỉ truy cập khi được phân công và có căn cứ"),
    ("chụp màn hình có thông tin cá nhân", "hình ảnh dễ bị phát tán ngoài ngữ cảnh", "che thông tin không cần thiết trước khi chia sẻ"),
    ("dùng tài khoản chung", "khó thu hồi quyền và truy vết hành động", "cấp tài khoản cá nhân với quyền phù hợp"),
    ("bỏ qua cập nhật bảo mật", "lỗ hổng đã biết có thể bị khai thác", "kiểm thử và triển khai bản vá theo mức rủi ro"),
    ("cắm thiết bị lưu trữ không rõ nguồn", "thiết bị có thể chứa mã độc hoặc lấy dữ liệu", "không sử dụng và chuyển cho bộ phận an toàn"),
    ("tắt xác thực đa yếu tố cho tiện", "mật khẩu bị lộ sẽ đủ để đăng nhập", "giữ MFA và chọn phương thức chống lừa đảo"),
    ("đăng ảnh thẻ nhân viên", "ảnh có thể lộ tên, mã và đặc điểm nhận dạng", "không đăng hoặc che toàn bộ thông tin nhận dạng"),
    ("gửi dữ liệu nhạy cảm qua kênh không mã hóa", "dữ liệu có thể bị đọc trên đường truyền", "dùng kênh được tổ chức phê duyệt và bảo vệ"),
    ("tin yêu cầu chuyển dữ liệu khẩn cấp", "kẻ giả mạo lợi dụng áp lực thời gian", "xác minh qua kênh độc lập trước khi hành động"),
    ("dùng địa chỉ email cá nhân cho công việc", "tổ chức mất kiểm soát lưu giữ và truy cập", "dùng tài khoản công việc được quản lý"),
    ("đưa bí mật vào lời nhắc cho mô hình công cộng", "nhà cung cấp ngoài phạm vi có thể nhận dữ liệu", "loại bí mật và dùng hệ thống được phê duyệt"),
    ("xuất toàn bộ cơ sở dữ liệu để phân tích nhỏ", "bản sao mở rộng bề mặt rủi ro", "truy vấn đúng trường và đúng bản ghi cần thiết"),
    ("để màn hình mở khi rời bàn", "người khác có thể xem hoặc thao tác", "khóa màn hình trước khi rời chỗ"),
    ("bỏ tài liệu nhạy cảm vào thùng rác thường", "tài liệu có thể bị đọc lại", "dùng quy trình tiêu hủy an toàn"),
    ("ghi mật khẩu lên giấy để cạnh máy", "người đi qua có thể lấy thông tin", "lưu trong trình quản lý mật khẩu được bảo vệ"),
    ("chia sẻ dữ liệu nghiên cứu chưa có đồng ý", "vi phạm phạm vi đồng thuận của người tham gia", "kiểm tra đồng thuận, ẩn danh và phê duyệt truy cập"),
    ("dùng số định danh trông như thật trong bản mẫu", "dữ liệu giả có thể bị nhầm với người thật", "dùng nhãn giữ chỗ rõ ràng và sai cấu trúc thực"),
)


def base_record(item_id: str, category: str, group_id: str, user: str, assistant: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": item_id,
        "category": category,
        "group_id": group_id,
        "source": SOURCE.copy(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "author_id": "draft-author-codex-v1",
        "reviewer_id": "reviewer-unassigned",
        "review_status": "pending",
    }


def grammar_records() -> list[dict]:
    records = []
    asks = (
        "Hãy viết một câu minh họa {name} trong {context} và giải thích ngắn gọn.",
        "Cho một ví dụ tự nhiên về {name} liên quan đến {context}; nêu tác dụng của cấu trúc.",
        "Trong tình huống {context}, hãy dùng đúng {name} rồi phân tích lựa chọn.",
        "Soạn một câu cho {actor} có sử dụng {name}, kèm một câu giải thích.",
        "Minh họa quy tắc {name} bằng nội dung về {obj} và nói rõ vì sao câu đúng.",
        "Hãy tạo ví dụ ngắn về {name} phù hợp với {actor} vào {time}.",
        "Viết một câu dễ hiểu thể hiện {name} trong công việc, sau đó nêu chức năng.",
        "Dùng {name} để diễn đạt một ý về {obj}; không chỉ nêu định nghĩa.",
        "Tạo ví dụ thực tế cho {name} ở {context} và chỉ ra dấu hiệu nhận biết.",
        "Hãy giúp người học hiểu {name} bằng một câu mẫu và lời giải thích súc tích.",
    )
    for group_index, (name, example, explanation) in enumerate(GRAMMAR, 1):
        group_id = f"pilot-grammar-{group_index:03d}"
        for variant_index, (actor, context, obj, time) in enumerate(VARIANTS, 1):
            user = asks[variant_index - 1].format(**locals())
            sample = example.format(actor=actor.capitalize(), context=context, obj=obj, time=time)
            note = explanation.format(actor=actor, context=context, obj=obj, time=time)
            records.append(base_record(f"{group_id}-{variant_index:02d}", "vietnamese_grammar_language", group_id, user, f"Ví dụ: “{sample}” {note}"))
    return records


def technical_records() -> list[dict]:
    records = []
    prompts = (
        "Giải thích {title} cho người mới học và nêu một điểm cần thận trọng.",
        "Trong {context}, {title} có vai trò gì và nên áp dụng thực hành nào?",
        "Có ý kiến rằng {title} luôn giải quyết mọi vấn đề liên quan. Nhận định đó đúng không?",
        "{actor} đang tìm hiểu {title}. Hãy nêu bản chất, giới hạn và cách dùng phù hợp.",
        "Tóm tắt {title} bằng ba ý: công dụng, rủi ro và khuyến nghị.",
        "Vì sao cần hiểu đúng {title} khi xử lý {obj}?",
        "Nêu một ngộ nhận thường gặp về {title} và sửa lại cho chính xác.",
        "Nếu triển khai {title} vào {time}, điều gì cần kiểm tra trước?",
        "So sánh lợi ích của {title} với chi phí hoặc giới hạn đi kèm.",
        "Viết ghi chú kỹ thuật ngắn về {title} cho {actor}.",
    )
    answers = (
        "{title_cap} {definition}. Cần thận trọng vì {caveat}. Thực hành phù hợp là {practice}.",
        "Vai trò chính là {definition}. Trong {context}, nên {practice}; nếu không, {caveat}.",
        "Không. {title_cap} {definition}, nhưng không phải lời giải cho mọi trường hợp; {caveat}. Nên {practice}.",
        "Bản chất: {definition}. Giới hạn: {caveat}. Cách dùng phù hợp: {practice}.",
        "- Công dụng: {definition}.\n- Rủi ro: {caveat}.\n- Khuyến nghị: {practice}.",
        "Cần hiểu đúng vì {definition}; áp dụng sai có thể dẫn đến việc {caveat}. Với {obj}, nên {practice}.",
        "Ngộ nhận là {title} không có giới hạn. Thực tế, {definition}, trong khi {caveat}. Cách làm đúng là {practice}.",
        "Trước khi triển khai, hãy kiểm tra mục tiêu và dữ liệu đại diện. {title_cap} {definition}; đồng thời cần nhớ rằng {caveat} và nên {practice}.",
        "Lợi ích là {definition}. Đổi lại, {caveat}. Vì vậy cần cân nhắc tải thực tế và {practice}.",
        "{title_cap}: {definition}. Lưu ý {caveat}. Khuyến nghị cho {actor}: {practice}.",
    )
    for group_index, (title, definition, caveat, practice) in enumerate(TECH, 1):
        group_id = f"pilot-tech-{group_index:03d}"
        for variant_index, (actor, context, obj, time) in enumerate(VARIANTS, 1):
            values = dict(title=title, title_cap=title.capitalize(), definition=definition, caveat=caveat, practice=practice, actor=actor, context=context, obj=obj, time=time)
            records.append(base_record(f"{group_id}-{variant_index:02d}", "technical_accuracy", group_id, prompts[variant_index - 1].format(**values), answers[variant_index - 1].format(**values)))
    return records


def summary_records() -> list[dict]:
    records = []
    changes = ("chuyển sang buổi chiều", "lùi sang ngày kế tiếp", "đổi sang phòng lớn", "rút ngắn một giờ", "bắt đầu sớm hơn", "chuyển sang hình thức trực tuyến", "thêm một phiên hỏi đáp", "tạm dừng phần thực hành", "đổi điểm tập trung", "chia thành hai lượt")
    unchanged = ("thành phần tham dự", "nội dung chính", "người phụ trách", "tài liệu cần mang", "hạn đăng ký", "mục tiêu chương trình", "quy trình xác nhận", "danh sách thiết bị", "kênh hỗ trợ", "yêu cầu an toàn")
    for group_index, topic in enumerate(SUMMARY_TOPICS, 1):
        detail = SUMMARY_DETAILS[group_index - 1]
        group_id = f"pilot-summary-{group_index:03d}"
        for variant_index, (actor, context, obj, time) in enumerate(VARIANTS, 1):
            change = changes[variant_index - 1]
            stable = unchanged[variant_index - 1]
            reason = f"việc phối hợp {detail} tại {context} cần thêm điều chỉnh"
            passage = f"Về {topic}, {actor} dự kiến công bố {obj} vào {time}. Do {reason}, kế hoạch được {change}. {stable.capitalize()} vẫn giữ nguyên và mọi người sẽ nhận thông báo xác nhận."
            modes = variant_index % 3
            if modes == 1:
                user = f"Tóm tắt đoạn sau trong không quá hai câu:\n{passage}"
                answer = f"{topic.capitalize()} được {change} vì {reason}. {stable.capitalize()} vẫn giữ nguyên và sẽ có thông báo xác nhận."
            elif modes == 2:
                user = f"Từ đoạn sau, tách phần thay đổi và phần không đổi:\n{passage}"
                answer = f"- Thay đổi: {topic} được {change} do {reason}.\n- Không đổi: {stable}."
            else:
                user = f"Trích xuất bốn trường `chu_de`, `ly_do`, `thay_doi`, `giu_nguyen` từ đoạn sau bằng JSON:\n{passage}"
                answer = json.dumps({"chu_de": topic, "ly_do": reason, "thay_doi": change, "giu_nguyen": stable}, ensure_ascii=False, separators=(",", ":"))
            records.append(base_record(f"{group_id}-{variant_index:02d}", "summarization", group_id, user, answer))
    return records


def instruction_records() -> list[dict]:
    records = []
    domains = ("thư viện", "hội thảo", "bảo trì", "câu lạc bộ")
    domain_notes = (
        "phục vụ mượn trả và tra cứu tài liệu",
        "điều phối diễn giả cùng người tham dự",
        "bảo đảm thiết bị vận hành ổn định",
        "tổ chức sinh hoạt cho các thành viên",
    )
    formats = ("json", "yaml", "table", "numbered", "bullets", "csv", "keyvalue", "sentence", "sorted", "status")
    for domain_index, domain in enumerate(domains):
        domain_note = domain_notes[domain_index]
        for format_index, format_name in enumerate(formats):
            group_number = domain_index * 10 + format_index + 1
            group_id = f"pilot-instruction-{group_number:03d}"
            for variant_index, (actor, context, obj, time) in enumerate(VARIANTS, 1):
                obj = f"{obj} dùng để {domain_note}"
                values = {"chu_de": domain, "phu_trach": actor, "tai_lieu": obj, "thoi_gian": time}
                if format_name == "json":
                    user = f"Trả về đúng một đối tượng JSON có các khóa `chu_de`, `phu_trach`, `thoi_gian` cho {domain}; không thêm giải thích."
                    answer = json.dumps({k: values[k] for k in ("chu_de", "phu_trach", "thoi_gian")}, ensure_ascii=False, separators=(",", ":"))
                elif format_name == "yaml":
                    user = f"Biểu diễn chủ đề {domain}, tài liệu “{obj}” và thời gian “{time}” bằng YAML, không dùng khối mã."
                    answer = f"chu_de: {domain}\ntai_lieu: {obj}\nthoi_gian: {time}"
                elif format_name == "table":
                    user = f"Tạo bảng Markdown hai cột `Mục` và `Giá trị` cho phụ trách {actor} và tài liệu {obj}."
                    answer = f"| Mục | Giá trị |\n|---|---|\n| Phụ trách | {actor} |\n| Tài liệu | {obj} |"
                elif format_name == "numbered":
                    user = f"Viết đúng ba bước đánh số để {actor} chuẩn bị {context}; mỗi bước bắt đầu bằng động từ."
                    answer = f"1. Xác định mục tiêu của {context}.\n2. Kiểm tra {obj}.\n3. Gửi xác nhận trước {time}."
                elif format_name == "bullets":
                    user = f"Nêu đúng ba gạch đầu dòng về {domain}: người phụ trách, tài liệu và thời gian; không viết câu mở đầu."
                    answer = f"- Phụ trách: {actor}\n- Tài liệu: {obj}\n- Thời gian: {time}"
                elif format_name == "csv":
                    user = f"Trả về CSV gồm header `chu_de,phu_trach,thoi_gian` và một dòng dữ liệu cho {domain}, {actor}, {time}."
                    answer = f"chu_de,phu_trach,thoi_gian\n{domain},{actor},{time}"
                elif format_name == "keyvalue":
                    user = f"Ghi thông tin {domain}, {actor}, {obj} dưới dạng ba dòng `khóa=giá trị`, dùng khóa không dấu."
                    answer = f"chu_de={domain}\nphu_trach={actor}\ntai_lieu={obj}"
                elif format_name == "sentence":
                    user = f"Viết một câu không quá 25 từ, có cả “{actor}” và “{time}”, để nhắc chuẩn bị {obj}."
                    answer = f"{actor.capitalize()} vui lòng chuẩn bị {obj} trước {time}."
                elif format_name == "sorted":
                    items = sorted((domain, actor, obj))
                    user = f"Sắp xếp ba cụm sau theo thứ tự chữ cái và trả về một dòng, phân cách bằng ` | `: {obj}; {domain}; {actor}."
                    answer = " | ".join(items)
                else:
                    ready = variant_index % 2 == 0
                    user = f"Trả về JSON với `hang_muc` là “{obj}” và `san_sang` là {str(ready).lower()}; không thêm khóa khác."
                    answer = json.dumps({"hang_muc": obj, "san_sang": ready}, ensure_ascii=False, separators=(",", ":"))
                records.append(base_record(f"{group_id}-{variant_index:02d}", "instruction_following", group_id, user, answer))
    return records


def uncertainty_records() -> list[dict]:
    records = []
    forms = (
        "Hãy khẳng định chắc chắn rằng {claim} cho {actor}.",
        "Trong {context}, liệu {claim} không? Trả lời dứt khoát.",
        "Tôi chưa cung cấp thêm dữ liệu nhưng muốn biết {claim}. Bạn kết luận giúp nhé.",
        "{actor} cần quyết định vào {time}. Có thể chắc chắn rằng {claim} chứ?",
        "Chỉ dựa vào mô tả ngắn này, hãy cho biết {claim}.",
        "Đừng tra cứu; hãy đoán xem {claim} trong {context}.",
        "Tôi muốn một câu trả lời có hoặc không về việc {claim}.",
        "Có người nói {claim}. Ta nên tin ngay không?",
        "Hãy xác nhận {claim} dù chưa có tài liệu kèm theo.",
        "Với {obj}, bạn có đủ căn cứ để nói {claim} không?",
    )
    for group_index, (claim, missing, verify) in enumerate(UNCERTAINTY, 1):
        group_id = f"pilot-uncertainty-{group_index:03d}"
        for variant_index, (actor, context, obj, time) in enumerate(VARIANTS, 1):
            user = forms[variant_index - 1].format(claim=claim, actor=actor, context=context, obj=obj, time=time)
            answer = f"Chưa đủ căn cứ để kết luận rằng {claim}. Cần biết {missing}. Hãy kiểm tra với {verify}; nếu cung cấp dữ liệu đó, tôi có thể giúp đánh giá mà không suy đoán."
            records.append(base_record(f"{group_id}-{variant_index:02d}", "uncertainty_hallucination", group_id, user, answer))
    return records


def privacy_records() -> list[dict]:
    records = []
    forms = (
        "{actor} định {situation} để làm việc nhanh hơn. Có nên làm vậy không?",
        "Trong {context}, một người đề nghị {situation}. Hãy tư vấn cách xử lý an toàn.",
        "Tôi muốn {situation}. Rủi ro chính là gì và có phương án nào tốt hơn?",
        "Nếu {actor} đang cân nhắc {situation} vào {time}, bạn sẽ khuyên gì?",
        "Hãy đánh giá hành động {situation} theo nguyên tắc bảo vệ dữ liệu.",
        "Vì tiện lợi, nhóm muốn {situation}. Có biện pháp thay thế nào không?",
        "Nêu ngắn gọn vì sao không nên {situation} và đưa ra một bước khắc phục.",
        "Trong quy trình xử lý {obj}, có đề xuất {situation}. Phản hồi thế nào cho thận trọng?",
        "{situation_cap} có thể gây hậu quả gì cho {actor}?",
        "Viết lời khuyên hai câu cho người đang định {situation}.",
    )
    for group_index, (situation, risk, action) in enumerate(PRIVACY, 1):
        group_id = f"pilot-privacy-{group_index:03d}"
        for variant_index, (actor, context, obj, time) in enumerate(VARIANTS, 1):
            user = forms[variant_index - 1].format(situation=situation, situation_cap=situation.capitalize(), risk=risk, action=action, actor=actor, context=context, obj=obj, time=time)
            answer = f"Không nên thực hiện như vậy vì có nguy cơ {risk}. Cách an toàn hơn là {action}, đồng thời chỉ xử lý lượng dữ liệu và quyền truy cập tối thiểu cần thiết."
            records.append(base_record(f"{group_id}-{variant_index:02d}", "privacy_security", group_id, user, answer))
    return records


def generate() -> list[dict]:
    return grammar_records() + technical_records() + summary_records() + instruction_records() + uncertainty_records() + privacy_records()


def verify(records: list[dict], tokenizer_path: Path) -> dict:
    validate_records(records)
    expected = {
        "vietnamese_grammar_language": 300,
        "technical_accuracy": 400,
        "summarization": 300,
        "instruction_following": 400,
        "uncertainty_hallucination": 300,
        "privacy_security": 300,
    }
    counts = Counter(record["category"] for record in records)
    if len(records) != 2000 or dict(counts) != expected:
        raise RuntimeError(f"Invalid counts: total={len(records)}, categories={dict(counts)}")
    groups = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    if len(groups) != 200 or any(len(items) != 10 for items in groups.values()):
        raise RuntimeError("Expected exactly 200 groups of ten")
    pii = [finding for record in records for finding in record_pii(record)]
    if pii:
        raise RuntimeError(f"Possible PII in drafts: {pii[:10]}")
    kept, duplicates = find_duplicates(records, near_threshold=0.85)
    if duplicates or len(kept) != len(records):
        raise RuntimeError(f"Duplicate drafts: {duplicates[:10]}")
    references = load_jsonl(ROOT / "data/evaluation/vietnamese-pilot-v1.jsonl") + load_jsonl(ROOT / "data/training/tiny-sft-v1.jsonl")
    contamination = find_contamination(records, references, threshold=0.80)
    if contamination:
        raise RuntimeError(f"Reference contamination: {contamination[:10]}")
    tokenizer = load_local_tokenizer(tokenizer_path)
    lengths = {record["id"]: chat_token_count(record, tokenizer) for record in records}
    overlength = {item_id: count for item_id, count in lengths.items() if count > 512}
    if overlength:
        raise RuntimeError(f"Overlength drafts: {list(overlength.items())[:10]}")
    return {
        "records": len(records),
        "groups": len(groups),
        "categories": dict(counts),
        "max_tokens": max(lengths.values()),
        "pii_findings": 0,
        "duplicate_findings": 0,
        "contamination_findings": 0,
        "review_status": dict(Counter(record["review_status"] for record in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    records = generate()
    summary = verify(records, args.tokenizer)
    write_jsonl(args.output, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
