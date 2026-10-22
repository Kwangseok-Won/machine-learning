"""Build the local ELI5 book from Python sources without importing the model."""
import ast
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAGES = [
    ('index.html', '전체 목차'),
    ('Transformer_Project_Overview_ELI5.html', '프로젝트 개요'),
    ('Transformer_AI_Reading_Guide.html', '사전 지식'),
    ('Transformer_Data_Flow_ELI5.html', '데이터 흐름'),
    ('Transformer_Encoder_Blocks_ELI5.html', '그림별 Encoder 해설'),
    ('Transformer_Code_Walkthrough_ELI5.html', '전체 함수 해설'),
    ('Transformer_AI_Code_Book.html', '전체 코드 교재'),
]
GROUPS = {
    1: '1 · 문장과 학습 재료 준비', 2: '2 · 단어 특징과 위치 표시',
    3: '3 · Attention과 계산 부품', 4: '4 · 그림의 Encoder',
    5: '5 · Decoder와 전체 Transformer', 6: '6 · 학습·평가·기록',
}

def S(match, title, before, after, text):
    return dict(match=match, title=title, before=before, after=after, text=text)

# Each explanation points to a literal line in the original source.
STEPS = {
 'TransformerEmbedding.forward': [
  S('tok_emb =', '단어 특징 카드를 꺼내요', '단어 번호 [B,L]', '단어별 숫자 512개 [B,L,512]', 'self.tok_emb(x)는 TokenEmbedding을 부릅니다. 번호로 표의 한 행을 골라 오며, 표의 숫자는 학습하면서 바뀝니다.'),
  S('pos_emb =', '단어 자리 표시를 꺼내요', '문장 길이 L', '위치별 숫자 [L,512]', 'PositionalEncoding은 단어 번호의 뜻을 보지 않고 길이에 맞는 자리 표시를 돌려줍니다.'),
  S('return self.drop_out', '특징과 자리를 같은 칸끼리 더해요', '특징 카드 + 자리 표시', '순서 정보가 포함된 특징', '그림의 ⊕입니다. 이어 붙이지 않고 같은 위치의 숫자를 더합니다. [L,512]는 배치의 모든 문장에 같은 방식으로 적용됩니다. 그 뒤 Dropout을 거칩니다.')],
 'PositionalEncoding.forward': [
  S('batch_size, seq_len', '들어온 문장의 길이를 알아요', '단어 번호 표 [B,L]', 'seq_len = L', 'x.size()에서 묶음 수와 단어 수를 읽습니다. 이 함수는 단어의 실제 값으로 위치를 만들지 않습니다.'),
  S('return self.encoding', '필요한 자리 표시만 잘라 줘요', '최대 256자리의 위치 표', '앞의 L자리 [L,512]', '생성자에서 sin·cos로 만든 고정 표를 잘라 돌려줍니다. 이 표는 학습으로 갱신하지 않습니다.')],
 'Encoder.forward': [
  S('x = self.emb', '그림 아래의 임베딩과 위치 더하기', '문장 번호 [B,L]', '특징과 순서 [B,L,512]', 'self.emb는 TransformerEmbedding입니다. TokenEmbedding → PositionalEncoding → 더하기 → Dropout을 한 번 실행합니다.'),
  S('for layer in', '큰 테두리 안을 여섯 층 통과해요', '이전 층이 만든 문장 특징', '다음 층이 가공한 문장 특징', 'self.layers에는 서로 다른 가중치를 가진 EncoderLayer 6개가 있습니다. 임베딩 전체를 여섯 번 반복하지 않습니다.'),
  S('x = layer(', '한 층의 Attention과 FFN을 실행해요', 'x와 src_mask', '한 EncoderLayer의 결과', 'layer(x, src_mask)는 PyTorch를 통해 EncoderLayer.forward를 호출합니다. 모양은 [B,L,512]로 유지됩니다.'),
  S('return x', '문장 메모를 밖으로 건네요', '마지막 EncoderLayer 결과', 'enc_src [B,L,512]', '아직 독일어 단어가 아닙니다. 다른 단어와의 관계가 반영된 숫자 메모이며 Decoder가 참고합니다.')],
 'EncoderLayer.forward': [
  S('_x = x', '옆길로 원래 카드를 보관해요', 'Attention 전 x', '_x에 원래 입력 보관', '그림의 첫 번째 우회 화살표입니다. 다음 계산이 끝난 뒤 새 결과에 원래 정보를 더할 준비를 합니다.'),
  S('x = self.attention', '같은 문장 안의 단어를 살펴요', '같은 x에서 출발한 Q·K·V', '관련 정보가 섞인 x', 'Self-Attention입니다. Q·K·V는 같은 입력으로 시작하지만 서로 다른 Linear 변환을 거칩니다. src_mask는 빈칸을 가립니다.'),
  S('x = self.norm1', '첫 번째 Add & Norm', 'Attention 결과 + 원래 입력', '숫자 크기를 정돈한 결과', '직전 줄에서 Dropout을 거친 x와 _x를 더합니다. norm1은 각 단어의 마지막 512개 특징 축을 정규화합니다.'),
  S('x = self.ffn', '단어별 특징을 가공해요', '관계 정보를 모은 카드', '512 → 2048 → 512', 'FFN은 각 단어 자리에 같은 가공기를 적용합니다. 단어끼리 새로 정보를 섞는 일은 Attention이 맡습니다.'),
  S('x = self.norm2', '두 번째 Add & Norm', 'FFN 결과 + FFN 전 입력', '한 EncoderLayer의 완성 결과', '이번 _x는 FFN 바로 전의 x입니다. Dropout → 더하기 → norm2를 거친 결과를 반환합니다.')],
 'MultiHeadAttention.forward': [
  S('q, k, v = self.w_q', '질문·표지·내용을 각각 만들어요', '입력 Q·K·V', '서로 다른 변환을 거친 Q·K·V', 'w_q, w_k, w_v는 별개의 학습 가능한 Linear입니다. 같은 문장에서 출발해도 변환 결과는 달라집니다.'),
  S('q, k, v = self.split', '여덟 관점으로 나눠요', '[B,L,512]', '[B,8,L,64]', '512개 특징을 8개 Head로 나눕니다. 문장을 여덟 조각으로 자르는 것이 아니라 각 단어의 특징 축을 나눕니다.'),
  S('out, attention =', '관점마다 단어 관계를 계산해요', '나눈 Q·K·V와 mask', '관계에 따라 모은 정보', 'ScaleDotProductAttention이 관련도 → 가림막 → Softmax → 내용 섞기를 수행합니다.'),
  S('out = self.concat', '여덟 결과를 다시 합쳐요', '[B,8,L,64]', '[B,L,512]', 'Head 축을 되돌려 붙인 뒤 w_concat으로 다시 가공합니다. attention 가중치를 시각화하는 기능은 TODO이며 이 함수는 out만 반환합니다.')],
 'MultiHeadAttention.split': [
  S('d_tensor =', '한 관점이 맡을 특징 수를 정해요', 'd_model=512, n_head=8', 'd_tensor=64', '특징 수가 Head 수로 나누어떨어지는 설정을 사용합니다.'),
  S('tensor = tensor.view', '숫자 표의 축을 정리해요', '[B,L,512]', '[B,8,L,64]', 'view로 Head 축을 만든 뒤 transpose로 Head를 두 번째 축으로 옮깁니다. 학습 가중치를 바꾸는 계산은 아닙니다.')],
 'MultiHeadAttention.concat': [
  S('d_model =', '합칠 특징 수를 계산해요', '8개 Head × 64개 특징', '512개 특징', 'split의 반대 방향으로 갈 준비를 합니다.'),
  S('tensor = tensor.transpose', 'Head 결과를 원래 모양으로 붙여요', '[B,8,L,64]', '[B,L,512]', 'transpose로 축을 되돌리고 contiguous로 메모리를 정리한 뒤 view로 Head와 특징 축을 합칩니다.')],
 'ScaleDotProductAttention.forward': [
  S('score = (q @ k_t)', '어느 단어와 관련 있는지 계산해요', 'Q와 K', '단어별 관련도 점수', 'Q와 K의 내적으로 관련도를 만들고 특징 수의 제곱근으로 나눠 값의 크기를 조절합니다.'),
  S('score = score.masked_fill', '보면 안 되는 칸을 가려요', '관련도 + mask', '가린 칸의 점수 −10000', 'mask가 0인 칸의 점수를 아주 작게 합니다. Encoder는 주로 패딩을 가리고 Decoder는 미래 단어도 가립니다.'),
  S('score = self.softmax', '얼마나 참고할지 비율을 정해요', '관련도 점수', '각 Query 행의 참고 비율', 'Softmax는 마지막 축에 적용됩니다. 각 Query가 여러 Key를 얼마나 참고할지 비율을 만듭니다.'),
  S('v = score @ v', '비율대로 내용을 섞어요', '참고 비율과 V', '새 내용과 참고 비율', '비율을 V에 곱해 필요한 정보를 모읍니다. return v, score로 내용과 가중치를 함께 반환합니다.')],
 'PositionwiseFeedForward.forward': [
  S('x = self.linear1', '특징 카드를 넓게 펼쳐요', '단어별 512개 숫자', '2048개 숫자', '같은 Linear를 모든 단어 자리에 적용해 특징을 더 넓은 공간에서 가공합니다.'),
  S('x = self.relu', '값을 비선형으로 가공해요', '펼친 특징', '음수는 0, 나머지는 유지', 'ReLU 덕분에 단순한 직선 변환 두 번보다 복잡한 관계를 학습할 수 있습니다. 뒤의 Dropout은 훈련할 때만 일부 값을 가립니다.'),
  S('x = self.linear2', '원래 카드 크기로 접어요', '2048개 특징', '512개 특징', '원래 입력과 더할 수 있게 마지막 특징 크기를 512로 되돌립니다.')],
 'LayerNorm.forward': [
  S('mean =', '카드 한 장의 평균을 구해요', '각 단어의 512개 특징', '단어별 평균', '마지막 축 −1만 사용합니다. 서로 다른 문장이나 단어의 통계를 하나로 섞지 않습니다.'),
  S('var =', '숫자가 퍼진 정도를 구해요', '각 단어의 특징', '단어별 분산', 'keepdim=True로 축의 자리를 남겨 이후 계산에서 원래 카드와 맞춰 계산합니다.'),
  S('out = (x - mean)', '크기를 정돈해요', '원래 특징과 평균·분산', '정규화한 특징', '평균을 빼고 표준편차로 나눕니다. eps는 0으로 나누는 문제를 줄이는 작은 값입니다.'),
  S('out = self.gamma', '학습 가능한 크기와 위치를 적용해요', '정규화한 특징', 'gamma × 특징 + beta', 'gamma와 beta는 학습 가능한 숫자입니다. 특징 크기와 반환 모양은 입력과 같습니다.')],
 'Transformer.forward': [
  S('src_mask =', '입력의 빈칸을 표시해요', 'src 번호 표', 'src_mask', '영어 문장의 패딩을 Attention에서 참고하지 않도록 준비합니다.'),
  S('trg_mask =', '정답의 미래를 가려요', 'trg 번호 표', 'trg_mask', '다음 단어를 맞힐 때 뒤에 있는 정답을 미리 보지 못하게 하는 삼각형 가림막을 만듭니다.'),
  S('enc_src =', '그림 전체를 실행해요', 'src와 src_mask', 'Encoder의 문장 메모', '사용자가 준 그림 전체가 self.encoder(...) 한 번에 해당합니다.'),
  S('output = self.decoder', '메모를 참고해 단어 후보 점수를 만들어요', 'trg + Encoder 메모 + 두 mask', '[B,T,독일어 어휘 수]', 'Decoder는 이 점수(logits)를 반환합니다. 이 함수 안에서 문장 문자열을 만들거나 Softmax로 단어 확률을 반환하지 않습니다.')],
 'Transformer.make_src_mask': [
  S('src_mask =', '패딩 번호만 제외해요', 'src [B,S]', 'mask [B,1,1,S]', 'src != src_pad_idx로 실제 단어 위치는 True, 패딩 위치는 False로 만듭니다. Head와 Query 축에 같은 가림막을 적용하도록 축을 두 개 추가합니다.')],
 'Transformer.make_trg_mask': [
  S('trg_pad_mask =', '패딩 여부를 표시해요', 'trg [B,T]', '[B,1,T,1]', '이 코드에서는 unsqueeze(3)을 사용하므로 Query 행 기준으로 패딩을 가리는 모양입니다. 구현 원문 그대로 설명합니다.'),
  S('trg_sub_mask =', '미래 단어를 못 보게 해요', 'T × T 표', '아래 삼각형만 1인 표', '현재 자리와 그 이전 자리는 보고 뒤의 자리는 못 보도록 torch.tril을 사용합니다.'),
  S('trg_mask =', '두 조건을 합쳐요', '패딩 표시와 미래 가림막', 'Decoder Self-Attention mask', '& 연산으로 두 조건이 모두 허용하는 위치만 남깁니다.')],
 'Decoder.forward': [
  S('trg = self.emb', '독일어 앞부분에 특징과 순서를 붙여요', '목표 문장 앞부분 [B,T]', '[B,T,512]', 'Encoder와 같은 구조의 임베딩을 쓰지만 별도 가중치를 가진 Decoder의 임베딩입니다.'),
  S('trg = layer(', '메모를 참고하며 여섯 층을 통과해요', '목표 특징과 Encoder 메모', '문맥이 반영된 목표 특징', '각 DecoderLayer는 가린 Self-Attention → Encoder 참고 → FFN을 수행합니다.'),
  S('output = self.linear', '독일어 후보마다 점수를 줘요', '각 자리의 512개 특징', '[B,T,독일어 어휘 수]', '마지막 Linear가 어휘마다 점수를 만듭니다. 실제 다음 단어 선택·반복 생성 루프는 이 프로젝트에 별도로 없습니다.')],
 'DecoderLayer.forward': [
  S('x = self.self_attention', '지금까지의 목표 단어끼리 살펴요', 'dec와 trg_mask', '목표 문장의 관계 정보', '미래 정답을 가린 Self-Attention입니다. 뒤에서 첫 Add & Norm을 적용합니다.'),
  S('x = self.enc_dec_attention', 'Encoder 메모에서 필요한 내용을 찾아요', 'Q는 목표 특징, K·V는 Encoder 메모', '원문을 참고한 목표 특징', 'Encoder-Decoder Attention입니다. 입력 길이 S와 목표 길이 T가 달라도 계산할 수 있습니다. 뒤에서 두 번째 Add & Norm을 적용합니다.'),
  S('x = self.ffn', '모은 정보를 가공해요', '원문을 참고한 특징', '가공한 특징', 'FFN 뒤에 세 번째 Add & Norm을 거쳐 같은 크기의 결과를 반환합니다.')],
 'DataLoader.make_dataset': [
  S("elif self.ext == ('.en', '.de')", '번역 방향을 고릅니다', 'ext=(.en,.de)', 'source=영어, target=독일어', '이 프로젝트 data.py의 설정은 영어 → 독일어입니다. 반대 확장자 순서도 처리하는 분기가 있습니다.'),
  S('train_data, valid_data', '문제·정답 쌍을 세 묶음으로 나눠 받아요', 'Multi30k와 두 Field 규칙', 'train, valid, test 데이터', 'Field는 토큰화·소문자화·시작 및 끝 기호·배치 축 규칙을 담습니다. 실제 자료는 Multi30k.splits가 준비합니다.')],
 'DataLoader.build_vocab': [
  S('self.source.build_vocab', '영어 단어 번호표를 만들어요', '훈련 데이터, min_freq=2', '영어 단어 ↔ 번호', '훈련 데이터에서 정해진 횟수 이상 나온 단어를 번호표에 담습니다. 특수 토큰도 포함됩니다.'),
  S('self.target.build_vocab', '독일어 번호표도 따로 만들어요', '동일한 훈련 데이터', '독일어 단어 ↔ 번호', '두 언어는 별도의 어휘표를 사용합니다. 검증·시험 데이터로 어휘표를 새로 만들지 않습니다.')],
 'DataLoader.make_iter': [
  S('train_iterator, valid_iterator', '길이를 고려해 묶어서 건네요', 'train, valid, test + batch_size', '세 BucketIterator', '비슷한 길이끼리 묶어 패딩 낭비를 줄이는 torchtext 반복자를 만듭니다. 기본 배치 크기는 128이며 마지막 묶음은 더 작을 수 있습니다.')],
 'Tokenizer.tokenize_en': [S('return [tok.text', '영어 문장을 조각내요', '영어 문자열', '토큰 문자열 목록', 'spaCy 영어 tokenizer가 문장부호를 포함해 조각을 나눕니다. 아직 단어 번호나 Tensor는 아닙니다.')],
 'Tokenizer.tokenize_de': [S('return [tok.text', '독일어 문장을 조각내요', '독일어 문자열', '토큰 문자열 목록', 'spaCy 독일어 tokenizer로 조각을 만듭니다. 번호로 바꾸는 일은 Field와 어휘표가 담당합니다.')],
 'count_parameters': [S('return sum', '배울 수 있는 숫자의 개수를 세요', 'model.parameters()', '학습 가능한 원소 수', 'requires_grad가 True인 파라미터의 numel()을 모두 합칩니다. 예측이나 학습을 수행하지 않습니다.')],
 'initialize_weights': [S("if hasattr", '행렬 형태의 가중치를 초기화해요', '모델의 모듈 m', '초기화된 weight', 'weight가 있고 2차원 이상이면 Kaiming 초기화를 적용합니다. model.apply가 하위 모듈을 순회하며 부릅니다. 모든 bias를 초기화하는 코드는 아닙니다.')],
 'train': [
  S('model.train()', '연습 모드로 바꿔요', 'model', 'Dropout이 동작하는 모델', '학습 모드를 켭니다. 이 호출만으로 가중치가 바뀌지는 않습니다.'),
  S('optimizer.zero_grad()', '지난 수정 메모를 지워요', '이전 배치의 gradient', '새 계산 준비', 'PyTorch는 gradient를 누적하므로 배치마다 이전 값을 비웁니다.'),
  S('output = model', '정답을 한 칸 어긋나게 보여줘요', 'src와 trg의 마지막 토큰 전까지', '다음 토큰 점수 [B,T,V]', 'trg[:, :-1]을 입력으로 주고 trg[:, 1:]을 정답으로 사용합니다. 이전 정답을 제공하는 teacher forcing입니다.'),
  S('loss = criterion', '후보 점수와 정답 번호를 비교해요', '펼친 점수 [B×T,V]와 정답 [B×T]', 'CrossEntropyLoss 한 값', 'contiguous().view로 배치와 단어 자리를 합쳐 채점기에 전달합니다. 낮을수록 정답에 더 좋은 점수를 준 것입니다.'),
  S('loss.backward()', '어느 숫자를 고칠지 계산해요', 'loss', '파라미터별 gradient', 'backward는 수정 방향을 계산합니다. 그 다음 clip_grad_norm_이 수정 방향의 크기를 제한합니다.'),
  S('optimizer.step()', '이제 모델의 숫자를 바꿔요', 'Adam과 gradient', '수정된 파라미터', '실제 가중치 갱신은 여기서 일어납니다. 모든 배치의 loss를 누적해 평균을 반환합니다.')],
 'evaluate': [
  S('model.eval()', '채점 모드로 바꿔요', 'model', 'Dropout이 꺼진 모델', '평가 때는 무작위 Dropout을 사용하지 않습니다.'),
  S('with torch.no_grad()', '수정 방향은 계산하지 않아요', '검증 배치', 'gradient 없이 예측', 'optimizer.step은 호출하지 않습니다. 단, 이 평가도 trg의 정답 앞부분을 입력으로 제공합니다.'),
  S('output = model', '정답 앞부분을 주고 후보 점수를 받아요', 'src와 trg[:, :-1]', '출력 점수', '새 문장을 시작 토큰부터 자유롭게 끝까지 생성하는 평가와 다릅니다.'),
  S('output_words = output[j].max', '각 자리에서 가장 높은 점수를 고릅니다', '어휘별 점수', '자리별 토큰 번호', 'max(dim=1)[1]은 가장 높은 점수의 인덱스를 고릅니다. 이 예측을 다음 입력으로 되먹이는 생성 루프는 아닙니다.'),
  S('bleu = get_bleu', '토큰 겹침을 채점하려고 해요', '예측 단어 목록과 정답 단어 목록', '현재 구현의 BLEU 값', '현재 호출은 단어 목록을 넘기는데 get_bleu는 zip의 각 항목을 문장처럼 처리합니다. 여기서는 단어 문자열이 다시 문자 단위 n-gram으로 계산되므로 표준 문장 BLEU로 해석하면 안 됩니다.'),
  S('return epoch_loss', '평균 검증 loss와 기록값을 돌려줘요', '배치별 loss와 BLEU', '평균 loss, 평균 BLEU', '단순 배치 평균이며 문장 수로 가중한 corpus BLEU가 아닙니다. test_iter가 아니라 run에서 넘긴 valid_iter를 사용합니다.')],
 'run': [
  S('train_loss = train', '문제집 한 바퀴를 연습해요', 'train_iter', '훈련 평균 loss', 'epoch마다 모든 훈련 배치를 처리합니다. 기본 설정은 1000 epoch입니다.'),
  S('valid_loss, bleu', '다른 문제로 점검해요', 'valid_iter', '검증 loss와 BLEU 기록', '이 구현의 검증은 정답 앞부분을 제공하는 방식입니다. test_iter는 준비되지만 이 run에서는 사용하지 않습니다.'),
  S('scheduler.step', '진전이 적으면 학습률을 조절해요', 'valid_loss', '필요하면 낮춘 학습률', 'step > warmup 조건 이후 ReduceLROnPlateau가 검증 loss의 개선을 관찰합니다. 초반 선형 warmup 코드는 아닙니다.'),
  S('torch.save', '지금까지 가장 잘한 상태를 저장해요', '이전 최저보다 작은 valid_loss', 'saved/model-숫자.pt', 'model.state_dict(), 즉 파라미터 상태를 저장합니다. optimizer나 어휘표 전체를 묶어 저장하는 코드는 아닙니다.'),
  S("f = open('result/train_loss", '학습 기록을 텍스트로 남겨요', '누적 loss·BLEU 목록', 'result 아래 세 파일', '파일을 w 모드로 다시 써서 지금까지의 목록을 남깁니다. test_loss.txt에는 실제로 검증 loss를 기록합니다. 출력 폴더가 존재해야 합니다.')],
 'idx_to_word': [S('word = vocab.itos', '단어 번호를 글자로 되돌려요', '토큰 번호 목록과 vocab', '단어 문자열', 'itos는 번호 → 단어 표입니다. < 문자가 든 토큰은 제외하고 공백으로 이어 붙입니다.')],
 'bleu_stats': [S('for n in range', '연속된 조각의 겹침을 세요', '후보와 정답의 토큰 시퀀스', '길이와 1~4-gram 통계 10개', 'Counter로 같은 조각의 개수를 세고 겹친 수를 모읍니다. 입력이 단어 목록이어야 단어 n-gram입니다. 문자열이면 문자를 조각으로 취급합니다.')],
 'bleu': [S('if len(list', '겹침이 없는 경우를 처리해요', '길이와 n-gram 통계', '0 또는 BLEU 비율', '어느 통계든 0이면 0을 반환합니다. 아니면 로그 평균 정밀도와 짧은 문장에 대한 벌점을 결합합니다. smoothing은 없습니다.')],
 'get_bleu': [S('for hyp, ref in zip', '후보·정답 문장 쌍의 통계를 모아요', '문장별 토큰 시퀀스의 목록 두 개', '0~100 척도의 점수', '각 hyp, ref를 bleu_stats에 넘긴 뒤 합산하고 bleu(stats)에 100을 곱합니다. evaluate에서는 중첩된 문장 목록 대신 단어 목록을 넘기므로 의도와 다르게 계산됩니다.')],
 'epoch_time': [S('elapsed_time =', '한 바퀴 걸린 시간을 정리해요', '시작·종료 시각', '분과 초의 두 정수', '초 단위 차이를 구해 몫은 분, 나머지는 초로 바꿉니다.')],
 'read': [S("file = f.read", '텍스트 기록을 숫자 목록으로 읽어요', '기록 파일 경로', '실수 목록', '대괄호를 제거하고 쉼표로 나눈 항목을 float로 바꿉니다. 학습 데이터가 아니라 저장한 지표를 읽는 함수입니다.')],
 'draw': [S("if mode == 'loss'", '어떤 기록을 그릴지 고릅니다', 'loss 또는 bleu', 'Matplotlib 그래프', 'loss 모드는 훈련·검증 기록을 함께 그리고 bleu 모드는 BLEU 기록을 그립니다. 마지막 plt.show()로 표시합니다. 모델을 다시 실행하지 않습니다.')],
}

CLASS_META = {
 'Tokenizer': ('문장을 단어 조각으로 나누는 가위', '영어·독일어 문자열', '토큰 문자열 목록', 'data.py에서 만들고 Field가 tokenize 함수를 호출', ['Tokenizer.tokenize_en','Tokenizer.tokenize_de']),
 'DataLoader': ('문제집과 단어 번호표를 준비하는 담당자', '언어 방향·토큰화 함수·특수 토큰', '데이터·어휘표·배치 반복자', 'data.py의 전역 준비 코드', ['DataLoader.make_dataset','DataLoader.build_vocab','DataLoader.make_iter']),
 'TokenEmbedding': ('단어 번호에 맞는 특징 카드를 꺼내는 표', '단어 번호 [B,L]', '학습 가능한 숫자 카드 [B,L,512]', 'TransformerEmbedding의 tok_emb(x)', []),
 'PositionalEncoding': ('몇 번째 자리인지 알려 주는 고정 표', '최대 길이·특징 수·입력 길이', '위치 표시 [L,512]', 'TransformerEmbedding의 pos_emb(x)', ['PositionalEncoding.forward']),
 'TransformerEmbedding': ('단어 특징과 자리 표시를 합치는 입구', '단어 번호 [B,L]', 'Dropout을 거친 특징 [B,L,512]', 'Encoder.forward와 Decoder.forward', ['TokenEmbedding','PositionalEncoding.forward']),
 'ScaleDotProductAttention': ('관련도만큼 필요한 정보를 모으는 계산기', 'Q·K·V와 mask', '섞은 내용과 Attention 가중치', 'MultiHeadAttention.forward', []),
 'MultiHeadAttention': ('여덟 관점으로 단어 관계를 살피는 팀', 'Q·K·V와 mask', '합친 특징 [B,Query길이,512]', 'EncoderLayer와 DecoderLayer', ['MultiHeadAttention.split','ScaleDotProductAttention.forward','MultiHeadAttention.concat']),
 'LayerNorm': ('단어 카드의 숫자 크기를 정돈하는 도구', '특징 [B,L,512]', '같은 모양의 정규화 특징', 'EncoderLayer와 DecoderLayer의 Add & Norm', []),
 'PositionwiseFeedForward': ('각 단어의 특징을 펼쳤다가 접는 가공기', '단어별 특징 512개', '가공한 특징 512개', 'EncoderLayer와 DecoderLayer의 ffn(x)', []),
 'EncoderLayer': ('그림의 큰 테두리 안에 있는 한 층', 'x와 src_mask', '문맥을 반영한 x [B,L,512]', 'Encoder.forward가 여섯 층을 차례로 호출', ['MultiHeadAttention.forward','LayerNorm.forward','PositionwiseFeedForward.forward']),
 'Encoder': ('입력 문장을 읽어 숫자 메모를 만드는 친구', 'src [B,S]와 mask', '문장 메모 [B,S,512]', 'Transformer.forward', ['TransformerEmbedding.forward','EncoderLayer.forward']),
 'DecoderLayer': ('앞부분과 원문 메모를 함께 보는 한 층', '목표 특징·Encoder 메모·두 mask', '가공한 목표 특징 [B,T,512]', 'Decoder.forward가 여섯 층을 차례로 호출', ['MultiHeadAttention.forward','PositionwiseFeedForward.forward','LayerNorm.forward']),
 'Decoder': ('원문 메모를 참고해 다음 단어 점수를 만드는 친구', 'trg·Encoder 메모·두 mask', '어휘별 점수 [B,T,V]', 'Transformer.forward', ['TransformerEmbedding.forward','DecoderLayer.forward']),
 'Transformer': ('읽는 친구와 쓰는 친구를 연결한 전체 모델', 'src와 trg의 단어 번호 표', '다음 토큰 후보별 점수 [B,T,V]', 'train.py의 train과 evaluate', ['Transformer.make_src_mask','Transformer.make_trg_mask','Encoder.forward','Decoder.forward']),
}
FUNCTION_META = {
 'count_parameters': ('학습 가능한 숫자 개수 세기','model','학습 가능한 파라미터 원소 수','train.py의 모델 준비 코드',[]),
 'initialize_weights': ('가중치 행렬의 출발값 준비','모듈 m','초기화된 weight; 반환값 없음','model.apply(initialize_weights)',[]),
 'train': ('한 바퀴 문제를 풀고 모델을 고치기','모델·훈련 반복자·Adam·채점기·clip','배치 평균 훈련 loss','run',['Transformer.forward']),
 'evaluate': ('모델을 고치지 않고 검증하기','모델·검증 반복자·채점기','평균 검증 loss와 현재 구현의 BLEU','run',['Transformer.forward','idx_to_word','get_bleu']),
 'run': ('연습 → 점검 → 저장을 epoch마다 연결','전체 epoch 수와 초기 best_loss','모델 가중치·loss/BLEU 기록 파일','train.py의 __main__',['train','evaluate','epoch_time']),
 'idx_to_word': ('번호표를 다시 단어로 읽기','토큰 번호와 어휘표','공백으로 연결한 문자열','evaluate',[]),
 'bleu_stats': ('연속된 토큰 조각의 겹침 세기','후보·정답 토큰 시퀀스','길이·n-gram 통계','get_bleu',[]),
 'bleu': ('겹침 통계를 BLEU 비율로 바꾸기','통계 10개','0~1 범위 비율','get_bleu',[]),
 'get_bleu': ('문장 쌍의 통계를 모아 점수 만들기','문장별 토큰 시퀀스 목록','100을 곱한 점수','evaluate',['bleu_stats','bleu']),
 'epoch_time': ('초를 분·초로 나누기','시작·종료 시각','분, 초','run',[]),
 'read': ('저장한 기록을 숫자 목록으로 읽기','파일 경로','실수 목록','draw',[]),
 'draw': ('학습 기록을 선 그래프로 보기','loss 또는 bleu','표시한 그래프; 반환값 없음','graph.py의 __main__',['read']),
}

def group_for(file):
    if file in ('conf.py','data.py') or file.startswith(('util/data_loader','util/tokenizer')): return 1
    if '/embedding/' in file: return 2
    if '/layers/' in file: return 3
    if file.endswith(('encoder.py','encoder_layer.py')): return 4
    if file.startswith('models/'): return 5
    return 6

def resolve_steps(source, start, specs):
    lines = source.split('\n')
    result = []
    for spec in specs:
        step = dict(spec)
        match = step.pop('match')
        matches = [i for i,line in enumerate(lines) if match in line]
        assert matches, (start, match)
        step['line'] = start + matches[0]
        result.append(step)
    return result

sources = {}
lessons = []
for path in sorted(ROOT.rglob('*.py')):
    if HERE in path.parents: continue
    file = path.relative_to(ROOT).as_posix()
    source = path.read_text(encoding='utf-8')
    if not source.strip(): continue
    sources[file] = source
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            nodes.append((node.name,node,node.name))
            nodes.extend((node.name+'.'+method.name,method,node.name) for method in node.body if isinstance(method,ast.FunctionDef))
        elif isinstance(node,ast.FunctionDef): nodes.append((node.name,node,None))
    for key,node,owner in nodes:
        snippet = '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
        is_class = isinstance(node,ast.ClassDef)
        if owner:
            title,given,result,caller,calls = CLASS_META[owner]
            if is_class:
                specs = [S('class '+owner, '이 부품의 설계도를 등록해요', given, title, 'class 정의만으로 입력 문장이 처리되지는 않습니다. 객체를 만들면 __init__으로 부품을 준비하고, 객체(x)를 호출하면 forward가 실행됩니다.')]
                calls = [owner+'.__init__'] + ([owner+'.forward'] if any(isinstance(n,ast.FunctionDef) and n.name=='forward' for n in node.body) else calls)
            elif node.name=='__init__':
                title = title+' · 부품 준비'
                given = '생성자 인자: '+', '.join(arg.arg for arg in node.args.args if arg.arg!='self')
                result = '계산 규칙과 하위 부품을 self에 보관; 반환값 없음'
                caller = {'Encoder':'Transformer.__init__','Decoder':'Transformer.__init__','EncoderLayer':'Encoder.__init__','DecoderLayer':'Decoder.__init__','TransformerEmbedding':'Encoder.__init__와 Decoder.__init__','TokenEmbedding':'TransformerEmbedding.__init__','PositionalEncoding':'TransformerEmbedding.__init__','MultiHeadAttention':'EncoderLayer.__init__와 DecoderLayer.__init__','LayerNorm':'EncoderLayer.__init__와 DecoderLayer.__init__','PositionwiseFeedForward':'EncoderLayer.__init__와 DecoderLayer.__init__','ScaleDotProductAttention':'MultiHeadAttention.__init__','Tokenizer':'data.py의 Tokenizer()','DataLoader':'data.py의 DataLoader()','Transformer':'train.py의 Transformer()'}[owner]
                specs = [S('def __init__', '모델을 만들 때 준비하는 부분이에요', given, '필요한 부품과 설정', 'self는 지금 만드는 객체입니다. __init__은 부품을 준비하며, 실제 입력 한 배치를 처리하는 forward와 역할이 다릅니다.')]
                if owner=='PositionalEncoding': specs.append(S('self.encoding[:, 0::2]', '고정된 자리 표를 만들어요', '자리 번호와 특징 번호', 'sin·cos 위치 표시', '짝수 특징에는 sin, 홀수 특징에는 cos 값을 채웁니다. requires_grad=False이므로 이 표는 학습으로 갱신하지 않습니다.'))
                elif owner=='TokenEmbedding': specs.append(S('super(TokenEmbedding', '단어별 특징 표를 만들어요', 'vocab_size와 d_model', '어휘 수 × 512 표', 'nn.Embedding을 상속합니다. forward를 직접 쓰지 않고 PyTorch의 표 조회 기능을 물려받습니다. padding_idx는 코드에서 1로 지정합니다.'))
                elif owner=='Encoder': specs.append(S('self.layers =', '독립된 여섯 층을 등록해요', 'n_layers=6', 'ModuleList 안의 EncoderLayer 6개', '리스트 안에서 EncoderLayer를 새로 만들므로 각 층의 가중치는 서로 다릅니다. self.emb는 이 반복 밖에서 한 번 만듭니다.'))
                elif owner=='Tokenizer': specs.append(S('self.spacy_de', '두 언어의 가위를 준비해요', 'spaCy 언어 모델 이름', '독일어·영어 tokenizer', '실제 Python 코드를 실행하려면 두 spaCy 언어 모델이 설치되어 있어야 합니다. 이 HTML 교재는 모델을 로드하지 않습니다.'))
            else:
                title = title+' · '+node.name
                specs = STEPS[key]
                if key=='MultiHeadAttention.split': given,result='[B,L,512]','[B,8,L,64]'
                if key=='MultiHeadAttention.concat': given,result='[B,8,L,64]','[B,L,512]'
                if key=='PositionalEncoding.forward': given,result='단어 번호 표 [B,L]','고정 위치 표 [L,512]'
                if key=='Transformer.make_src_mask': given,result,caller='src [B,S]','패딩을 제외하는 mask [B,1,1,S]','Transformer.forward'
                if key=='Transformer.make_trg_mask': given,result,caller='trg [B,T]','목표 패딩·미래 가림막 [B,1,T,T]','Transformer.forward'
                if key in ('MultiHeadAttention.split','MultiHeadAttention.concat'): caller='MultiHeadAttention.forward'
                if key.startswith('DataLoader.'):
                    given,result={'make_dataset':('언어 방향과 Field 규칙','train, valid, test 데이터'), 'build_vocab':('훈련 데이터와 최소 빈도','source·target의 vocab; 반환값 없음'), 'make_iter':('세 데이터 묶음·batch_size·device','세 BucketIterator')}[node.name]
        else:
            title,given,result,caller,calls = FUNCTION_META[key]
            specs = STEPS[key]
        lessons.append(dict(key=key,file=file,start=node.lineno,source=snippet,group=GROUPS[group_for(file)],title=title,given=given,result=result,caller=caller,calls=list(calls),steps=resolve_steps(snippet,node.lineno,specs),kind='클래스' if is_class else '메서드' if owner else '함수'))

definition_count = len(lessons)
assert definition_count == 60, definition_count

def chunk(file,first,last):
    return '\n'.join(sources[file].splitlines()[first-1:last])

def add_module(key,file,first,last,title,given,result,caller,specs,calls):
    source = chunk(file,first,last)
    lessons.append(dict(key=key,file=file,start=first,source=source,group=GROUPS[group_for(file)],title=title,given=given,result=result,caller=caller,calls=calls,steps=resolve_steps(source,first,specs),kind='실행 구간'))

add_module('conf.py · 설정','conf.py',6,29,'모델 크기와 연습 규칙을 정해요','torch와 설정 값','전역 설정 변수','data.py가 import',[
 S('device =','계산할 장치를 골라요','CUDA 사용 가능 여부','GPU 또는 CPU','실제 CUDA 사용 가능 여부를 확인해 device를 정합니다.'),
 S('d_model =','그림의 크기를 정해요','512·6·8·2048','특징 512, 층 6, Head 8, FFN 2048','이 값들이 Embedding·N×·Multi-Head·FFN에 연결됩니다. 배치 128과 최대 위치 길이 256도 설정합니다.'),
 S('init_lr =','학습 규칙을 준비해요','학습률 1e-5 등','Adam과 scheduler 설정','설명은 README의 예시 숫자가 아닌 현재 conf.py 원문을 기준으로 합니다.')],[])
add_module('data.py · 준비','data.py',6,len(sources['data.py'].splitlines()),'import할 때 문장 쌍과 번호표를 준비해요','conf와 Tokenizer·DataLoader','어휘 수·특수 토큰 번호·세 반복자','train.py의 from data import *',[
 S('tokenizer =','문장을 조각낼 준비','spaCy tokenizer','Tokenizer 객체','모듈의 전역 코드는 import 시 실행됩니다.'),
 S('train, valid, test','문장 쌍을 가져와요','영어 → 독일어','세 데이터 묶음','Multi30k 자료를 읽는 준비가 여기서 시작됩니다.'),
 S('loader.build_vocab','훈련 자료로 번호표를 만들어요','훈련 문장','두 언어의 vocab','최소 등장 빈도 2를 사용합니다.'),
 S('src_pad_idx =','모델에 필요한 번호를 기록해요','두 vocab','패딩·시작 번호와 어휘 수','stoi는 단어 → 번호, itos는 번호 → 단어입니다.')],['Tokenizer','DataLoader.make_dataset','DataLoader.build_vocab','DataLoader.make_iter'])
add_module('train.py · 모델 준비','train.py',27,52,'새 모델과 채점·수정 도구를 조립해요','data.py가 준비한 어휘와 conf 설정','Transformer·Adam·scheduler·criterion','train.py 모듈 전역 코드',[
 S('model = Transformer','읽는 친구와 쓰는 친구를 조립해요','어휘 수와 모델 설정','device에 놓인 Transformer','객체를 만든 뒤 train 함수가 입력 배치를 전달합니다.'),
 S('model.apply','출발 가중치를 준비해요','모델의 하위 모듈','초기화된 weight','initialize_weights를 각 하위 모듈에 적용합니다.'),
 S('optimizer = Adam','숫자를 고칠 도구를 준비해요','모델 파라미터','Adam optimizer','optimizer.step이 불릴 때 실제 숫자가 바뀝니다.'),
 S('criterion =','틀린 정도를 잴 채점기예요','후보 점수와 정답','CrossEntropyLoss','원문은 ignore_index=src_pad_idx를 사용합니다. 목표 패딩 번호와 같다는 가정에 의존하며 여기서는 코드를 수정하지 않습니다.')],['Transformer','initialize_weights','train','evaluate'])
main_line = next(i+1 for i,line in enumerate(sources['train.py'].splitlines()) if line.startswith("if __name__"))
add_module('train.py · 시작점','train.py',main_line,len(sources['train.py'].splitlines()),'직접 실행할 때 전체 연습을 시작해요','epoch와 inf','run 호출','python train.py',[
 S("if __name__",'직접 실행했는지 확인해요','현재 모듈 이름','__main__일 때 run 실행','import만 할 때 이 조건 아래 run은 실행하지 않습니다. 다만 data 준비와 모델 생성 등 위쪽 전역 코드는 import 시에도 실행됩니다.')],['run'])
lessons.sort(key=lambda item:(item['group'],item['file'],item['start']))
by_key = {item['key']:item for item in lessons}

def block(key,label,file,first,last,title,given,result,caller,specs,calls,shape,rects,paths=None):
    source=chunk(file,first,last)
    return dict(key=key,label=label,file=file,start=first,source=source,title=title,given=given,result=result,caller=caller,steps=resolve_steps(source,first,specs),calls=calls,shape=shape,rects=rects,paths=paths or [])

blocks = [
 block('inputs','① Inputs · 입력','models/model/transformer.py',40,45,'단어 번호로 바꾼 영어 문장을 읽는 친구에게 보내요','src [B,S], src_mask','Encoder의 문장 메모','train → model(src, trg[:, :-1])', [STEPS['Transformer.forward'][0],STEPS['Transformer.forward'][2]],['Transformer.forward','Encoder.forward'],'입력: [묶음 B, 영어 단어 수 S]',[[163,451,75,25]]),
 block('embedding','② Input Embedding · 특징 카드','models/embedding/transformer_embedding.py',25,33,'단어 번호 하나마다 512개의 숫자 카드를 꺼내요','x [B,S]','tok_emb [B,S,512]','Encoder.forward → self.emb(x)',[S('self.tok_emb =','특징 카드 보관함을 만들어요','어휘 수와 특징 수 512','TokenEmbedding','nn.Embedding을 상속한 TokenEmbedding이 단어별 학습 가능한 표를 만듭니다.'),STEPS['TransformerEmbedding.forward'][0]],['TokenEmbedding','TokenEmbedding.__init__','TransformerEmbedding.forward'],'[B,S] → [B,S,512]',[[142,364,118,47]]),
 block('position','③ Positional Encoding + ⊕ · 자리 표시','models/embedding/transformer_embedding.py',30,33,'어떤 단어인지에 몇 번째 자리인지를 더해요','tok_emb [B,S,512]와 pos_emb [S,512]','순서가 담긴 특징 [B,S,512]','Encoder.forward → TransformerEmbedding.forward',STEPS['TransformerEmbedding.forward'][1:],['PositionalEncoding.__init__','PositionalEncoding.forward'],'512개 특징 + 512개 위치 표시 = 512개 특징',[[38,307,137,50],[191,324,20,20]]),
 block('attention','④ Multi-Head Attention · 관계 찾기','models/layers/multi_head_attention.py',22,39,'여덟 관점으로 어떤 단어 정보를 참고할지 계산해요','같은 문장의 Q·K·V와 src_mask','문맥이 반영된 특징 [B,S,512]','EncoderLayer.forward → self.attention(...)',STEPS['MultiHeadAttention.forward'],['EncoderLayer.forward','MultiHeadAttention.split','ScaleDotProductAttention.forward','MultiHeadAttention.concat'],'[B,S,512] → [B,8,S,64] → [B,S,512]',[[142,207,118,48]]),
 block('add1','⑤ Add & Norm · 첫 번째 옆길','models/blocks/encoder_layer.py',25,32,'원래 정보와 Attention 결과를 합쳐 숫자를 정돈해요','Attention 전 x와 처리 결과','norm1을 거친 특징','EncoderLayer.forward',STEPS['EncoderLayer.forward'][:3],['EncoderLayer.forward','LayerNorm.forward'],'Add: x + _x · Norm: 각 단어의 512개 특징 축',[[142,178,118,24]],['M201 287 H112 V190 H142']),
 block('ffn','⑥ Feed Forward · 카드 가공','models/layers/position_wise_feed_forward.py',11,23,'각 단어의 특징을 넓게 펼쳐 가공한 뒤 접어요','x [B,S,512]','가공한 x [B,S,512]','EncoderLayer.forward → self.ffn(x)',STEPS['PositionwiseFeedForward.forward'],['PositionwiseFeedForward.forward'],'512 → 2048 → ReLU·Dropout → 512',[[142,88,118,47]]),
 block('add2','⑦ Add & Norm · 두 번째 옆길','models/blocks/encoder_layer.py',34,41,'이번에는 FFN 전 정보를 새 결과에 보태요','FFN 전 _x와 FFN 결과','한 층의 최종 특징 [B,S,512]','EncoderLayer.forward',[S('_x = x','FFN 전 카드를 보관해요','첫 Add & Norm의 결과','이번 옆길의 _x','앞의 _x와 이름은 같지만 여기서 새 값으로 다시 보관합니다.'),STEPS['EncoderLayer.forward'][3],STEPS['EncoderLayer.forward'][4]],['EncoderLayer.forward','LayerNorm.forward'],'FFN 전 입력 + FFN 결과 → norm2',[[142,59,118,24]],['M201 159 H112 V70 H142']),
 block('repeat','⑧ N× · 큰 테두리 여섯 층','models/model/encoder.py',14,34,'큰 테두리 안의 ④~⑦만 독립된 여섯 층으로 이어 붙여요','n_layers=6, 임베딩 결과','여섯 층을 거친 [B,S,512]','Transformer.forward → Encoder.forward',[
 S('self.layers =','각 층의 부품을 따로 만들어요','EncoderLayer와 n_layers=6','서로 다른 EncoderLayer 6개','같은 부품을 재사용하지 않고 새 EncoderLayer를 여섯 개 만들고 ModuleList에 등록합니다.'),*STEPS['Encoder.forward'][:3]],['Encoder.__init__','Encoder.forward','EncoderLayer.forward'],'Embedding과 위치 표시는 한 번 · ④~⑦은 6층',[[97,47,176,254]]),
 block('output','⑨ 출력 화살표 · 문장 메모','models/model/transformer.py',40,45,'Encoder의 숫자 메모를 Decoder에게 건네요','Encoder의 마지막 특징','Decoder의 단어 후보 점수','Transformer.forward',[STEPS['Transformer.forward'][2],STEPS['Transformer.forward'][3]],['Encoder.forward','Decoder.forward','Transformer.forward'],'Encoder [B,S,512] → Decoder 참고 자료',[[187,25,28,34]])
]

def escape(value): return html.escape(str(value),quote=True)
def route(key,label=None): return '<a href="Transformer_Code_Walkthrough_ELI5.html#fn='+escape(key)+'">'+escape(label or key)+'</a>'

def page(filename,title,content,viewer=''):
    index=[p[0] for p in PAGES].index(filename)
    options=''.join('<option value="'+f+'"'+(' selected' if f==filename else '')+'>'+label+'</option>' for f,label in PAGES)
    prevnext=('' if index==0 else '<a href="'+PAGES[index-1][0]+'">← '+PAGES[index-1][1]+'</a>')+('' if index==len(PAGES)-1 else '<a href="'+PAGES[index+1][0]+'">'+PAGES[index+1][1]+' →</a>')
    output='<!doctype html>\n<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+' · Transformer ELI5</title><link rel="stylesheet" href="book.css"></head><body data-viewer="'+viewer+'"><header class="study-nav" aria-label="학습 페이지 이동"><div><a href="index.html"><b>전체 목차</b></a><label for="study-page" class="sr-only">현재 페이지</label><select id="study-page">'+options+'</select></div><div class="prevnext">'+prevnext+'</div></header><main>'+content+'<footer class="footer small">현재 Python 소스를 읽어 만든 설명 교재입니다. HTML 열람은 모델 학습이나 Python 코드를 실행하지 않습니다. 줄 번호는 원본 .py 파일 기준입니다.</footer></main><script src="book-data.js"></script><script src="book.js"></script></body></html>'
    (HERE/filename).write_text(output,encoding='utf-8')

def viewer(kind):
    picture=''
    if kind=='blocks':
        picture='<div class="reference-wrap"><figure class="reference"><img src="assets/encoder-reference.png" width="380" height="488" alt="사용자가 제공한 Encoder 그림. 아래에서 위로 입력, 임베딩, 위치 더하기, Attention, Add and Norm, Feed Forward, Add and Norm. 큰 테두리를 N번 반복."><svg viewBox="0 0 380 488" aria-hidden="true"><g id="highlight"></g></svg></figure><div class="small" id="block-position" aria-live="polite"></div><div class="shape" id="shape"></div><p class="small">주황색 테두리·선 = 현재 설명하는 블록·옆길</p></div>'
    main='<article class="panel step-card"><div class="small break" id="where"></div><h2 class="break" id="name"></h2><p id="purpose"></p><div class="contract"><div><b>받는 것</b><br><span id="given"></span></div><div><b>돌려주거나 남기는 것</b><br><span id="result"></span></div></div><p class="small" id="caller"></p>'
    main+='<div class="layout"><div><div class="small">실제 Python 원문 · 밝은 줄이 현재 설명</div><pre class="code" id="code" aria-label="실제 소스 코드"></pre><p class="small">왼쪽 숫자는 원본 파일의 줄 번호입니다. 원문과 들여쓰기를 보존했습니다.</p></div><div aria-live="polite"><h3 id="step-title"></h3><div class="diagram"><div class="box" id="before"></div><div class="arrow" aria-hidden="true">↓</div><div class="box after" id="after"></div></div><p id="explanation"></p></div></div>'
    main+='<nav class="controls" aria-label="설명 단계"><div><button id="prev">← 이전 설명</button><button id="next">다음 설명 →</button></div><span id="position" aria-live="polite"></span></nav><h3>연결된 코드로 가기</h3><div class="pills" id="calls"></div><nav class="controls" aria-label="항목 이동"><button id="prev-item">← 이전 '+('블록' if kind=='blocks' else '항목')+'</button><span class="small" id="item-count"></span><button id="next-item">다음 '+('블록' if kind=='blocks' else '항목')+' →</button></nav></article>'
    if picture: main='<div class="diagram-layout">'+picture+'<div>'+main+'</div></div>'
    return '<div class="workspace"><aside><details class="catalog" open><summary><b>'+('그림의 블록 목차' if kind=='blocks' else '함수·클래스·실행 구간')+'</b></summary><label class="small" for="filter">이름 찾기</label><input id="filter" type="search" placeholder="예: Attention, forward"><nav id="catalog" aria-label="항목 목차"></nav></details></aside><div>'+main+'</div></div>'

def index_content():
    cards=''.join('<a class="pagecard" href="'+filename+'"><b>'+label+'</b><span>'+description+'</span></a>' for (filename,label),description in zip(PAGES[1:],[
      '영어 → 독일어 번역 모델의 전체 역할과 파일 지도','Tensor·shape·class·forward·Loss를 쉬운 비유로','문장 한 쌍이 토큰·숫자·배치·점수로 바뀌는 과정','첨부 그림의 9개 위치와 원문 코드를 나란히 대조','60개 정의와 4개 실행 구간의 입력·처리·결과','학습·모델 코드의 비어 있지 않은 .py 파일 원문']))
    groups=''
    for group in GROUPS.values():
        rows=''.join('<a href="Transformer_Code_Walkthrough_ELI5.html#fn='+escape(item['key'])+'"><b>'+escape(item['key'])+'</b><span>'+escape(item['title'])+'</span></a>' for item in lessons if item['group']==group)
        groups+='<section class="group"><h2>'+group+'</h2><div class="functionlist">'+rows+'</div></section>'
    return '<div class="small">Transformer 번역 프로젝트 · 그림과 실제 코드로 배우는 교재</div><h1>그림 속 블록이<br>코드에서 어떻게 움직일까요?</h1><p class="page-intro">읽는 친구 <b>Encoder</b>가 영어 문장을 숫자 메모로 바꾸고, 쓰는 친구 <b>Decoder</b>가 그 메모를 보며 독일어 다음 단어의 점수를 만듭니다. 사용자가 제공한 Encoder 그림부터 프로젝트 전체 코드까지 연결했습니다.</p><div class="route">큰 흐름 이해 → 문장 한 쌍 따라가기 → 그림의 블록과 코드 대조 → 함수 안으로 들어가기</div><h2>페이지 바로가기</h2><div class="cards">'+cards+'</div><div class="note"><b>먼저 보려면 ‘그림별 Encoder 해설’을 여세요.</b><br>① 블록 선택 → ② 원본 그림의 강조 위치 확인 → ③ 밝은 코드 줄과 입력·결과 비교 → ④ 다음 설명 → ⑤ 연결 함수로 이동. 선택한 항목과 설명 단계는 URL에 남아 뒤로·앞으로 이동할 수 있습니다.</div><h2>모든 정의와 실행 구간 바로가기</h2><p class="small">클래스·생성자·메서드·함수 '+str(definition_count)+'개를 빠짐없이 수록했습니다. 함수 밖에서 실행되는 설정·데이터 준비·모델 조립·시작점 4곳도 포함합니다.</p>'+groups

page('index.html','전체 목차',index_content())
page('Transformer_Encoder_Blocks_ELI5.html','그림별 Encoder 해설','<div class="small">원본 그림 ↔ 실제 코드 · 아래에서 위로</div><h1>그림의 각 블록을 코드와 맞춰 읽어요.</h1><div class="note">이 그림 전체는 <b>Encoder</b>입니다. 그림 위쪽 출력도 번역 문장이 아니라 Decoder에게 줄 숫자 메모입니다. <b>N×는 큰 테두리 안의 층만 6번</b> 이어 붙인다는 뜻입니다. 아래 목차에서 블록을 고르면 그림과 실제 코드 줄이 함께 강조됩니다.</div>'+viewer('blocks'),'blocks')
page('Transformer_Code_Walkthrough_ELI5.html','전체 함수 해설','<div class="small">실제 .py 원문 · 입력 → 처리 → 결과</div><h1>60개 정의와 4개 실행 구간을<br>하나씩 따라가요.</h1><div class="flow">'+route('data.py · 준비','데이터 준비')+' → '+route('Transformer.forward','전체 모델')+' → '+route('Encoder.forward','Encoder')+' → '+route('Decoder.forward','Decoder')+' → '+route('train','연습')+' → '+route('evaluate','점검')+' → '+route('run','반복·저장')+'</div><p class="small">__init__은 부품 준비, forward는 입력 처리입니다. class 정의만으로 학습이 시작되지는 않습니다.</p>'+viewer('functions'),'functions')

page('Transformer_Project_Overview_ELI5.html','프로젝트 개요','''
<div class="small">먼저 숲을 보기</div><h1>읽는 친구와 쓰는 친구가<br>함께 번역을 배워요.</h1>
<div class="panel"><div class="diagram"><div class="box">영어 문장 + 독일어 정답 쌍</div><div class="arrow">↓</div><div class="box">Encoder · 영어를 읽고 숫자 메모 만들기</div><div class="arrow">↓ 메모를 참고</div><div class="box after">Decoder · 독일어 앞부분에서 다음 단어 맞히기</div><div class="arrow">↓</div><div class="box">Loss → backward → Adam.step → 반복</div></div></div>
<section class="panel"><h2>사용자가 준 그림은 어디인가요?</h2><div class="layout"><img class="wide-image" src="assets/encoder-reference.png" width="380" height="488" alt="입력에서 위로 올라가는 Encoder 원본 그림"><div><p>이 그림 전체가 <code>self.encoder(src, src_mask)</code>입니다. Encoder 안에는 임베딩·위치 표시와 여섯 EncoderLayer가 있습니다.</p><p>그림에는 Decoder와 Loss가 나오지 않습니다. Encoder 출력은 단어마다 512개의 숫자가 담긴 메모이고, 이 메모가 Decoder에 들어갑니다.</p><p><a href="Transformer_Encoder_Blocks_ELI5.html">그림별 코드 해설로 이동 →</a></p></div></div></section>
<section class="panel"><h2>파일별 역할</h2><table><thead><tr><th>파일·폴더</th><th>맡은 일</th></tr></thead><tbody>
<tr><td>conf.py</td><td>특징 512, 층 6, Head 8, FFN 2048, 배치 128 등 설정</td></tr><tr><td>data.py · util/tokenizer.py · util/data_loader.py</td><td>영어·독일어 쌍, 단어 조각, 번호표, 배치 준비</td></tr><tr><td>models/embedding</td><td>단어 번호를 숫자 특징으로 바꾸고 자리 표시 더하기</td></tr><tr><td>models/layers</td><td>Attention·FFN·정규화라는 작은 계산 부품</td></tr><tr><td>models/blocks</td><td>작은 부품을 조립한 EncoderLayer·DecoderLayer 한 층</td></tr><tr><td>models/model</td><td>여섯 층짜리 Encoder·Decoder와 전체 Transformer</td></tr><tr><td>train.py</td><td>모델 조립, 학습, 검증, 좋은 가중치 저장</td></tr><tr><td>util/bleu.py · graph.py</td><td>기록용 점수 계산과 학습 곡선 표시</td></tr></tbody></table></section>
<section class="panel"><h2>원문에서 확인한 실행 범위</h2><p><code>train.py</code>의 마지막 줄이 <code>run(total_epoch=epoch, best_loss=inf)</code>를 호출합니다. 학습은 train_iter, 중간 점검은 valid_iter를 사용합니다. test_iter는 준비되지만 이 run에서 사용하지 않습니다.</p><p>평가에도 정답 앞부분을 제공하며, 새 문장을 시작 토큰부터 끝까지 생성하는 별도 번역 루프는 없습니다. 교재는 현재 동작을 설명하며 원본 학습 코드를 바꾸지 않습니다.</p></section>
''')

page('Transformer_AI_Reading_Guide.html','사전 지식','''
<div class="small">어려운 단어를 쉬운 그림으로</div><h1>코드를 읽기 전에<br>이 여덟 가지만 알아두세요.</h1>
<div class="cards">
<section class="panel"><h2>① Tensor = 숫자가 들어 있는 상자</h2><div class="box">문장 묶음 → 단어 자리 → 특징 숫자</div><p><code>[B,L,D]</code>는 묶음 수 B, 단어 자리 수 L, 특징 수 D입니다. 이 모델의 D는 512입니다.</p></section>
<section class="panel"><h2>② shape = 상자의 칸 수</h2><div class="box">[2, 5, 512]</div><p>문장 두 개, 문장마다 다섯 자리, 자리마다 숫자 512개라는 예시입니다. 실제 어휘 번호나 학습 결과는 아닙니다.</p></section>
<section class="panel"><h2>③ class와 __init__</h2><div class="box">설계도 → 부품 조립</div><p>class는 부품을 만드는 규칙, __init__은 부품을 처음 만들 때 설정과 하위 도구를 준비하는 곳입니다. <code>self</code>는 지금 만든 부품 자신입니다.</p></section>
<section class="panel"><h2>④ forward = 재료가 지나가는 길</h2><div class="box">model(입력) → forward → 결과</div><p>PyTorch의 nn.Module 객체를 함수처럼 부르면 forward가 실행됩니다. <code>x = self.emb(x)</code>도 같은 원리입니다.</p></section>
<section class="panel"><h2>⑤ Embedding = 단어 특징 카드</h2><div class="box">단어 번호 → 숫자 512개</div><p>학습 가능한 표에서 단어 번호에 맞는 행을 가져옵니다. 초기 숫자에 사람이 미리 의미 이름을 붙여 둔 것은 아닙니다.</p></section>
<section class="panel"><h2>⑥ mask = 보지 않을 칸 표시</h2><div class="box">패딩 가리기 · 미래 정답 가리기</div><p>Encoder는 입력의 패딩을 무시합니다. Decoder의 Self-Attention은 뒤에 있는 목표 정답도 미리 볼 수 없도록 합니다.</p></section>
<section class="panel"><h2>⑦ Loss와 gradient</h2><div class="box">틀린 정도 → 고칠 방향 → 실제 수정</div><p>Loss가 틀린 정도를 재고, backward가 수정 방향을 구하고, optimizer.step이 모델 파라미터를 바꿉니다.</p></section>
<section class="panel"><h2>⑧ Dropout과 평가 모드</h2><div class="box">연습 중 일부 가리기 → 평가 때 모두 사용</div><p>Dropout은 학습 중 일부 값을 무작위로 0으로 만듭니다. model.eval()에서는 이 가리기를 끕니다. no_grad()는 gradient 기록을 끄는 별도 기능입니다.</p></section></div>
<section class="panel check"><h2>읽었는지 확인하기</h2><details><summary>Embedding은 숫자 512개를 8개 Head로 나누는 블록인가요?</summary><p>아니요. Embedding은 단어 번호를 특징으로 바꿉니다. Head 분리는 MultiHeadAttention.split의 역할입니다.</p></details><details><summary>N=6이면 위치 표시도 여섯 번 더하나요?</summary><p>아니요. 임베딩과 위치 더하기는 한 번, EncoderLayer 내부 계산만 여섯 층을 거칩니다.</p></details><details><summary>Encoder의 결과는 독일어 문장인가요?</summary><p>숫자 메모입니다. Decoder가 이 메모를 참고해 다음 단어 후보의 점수를 만듭니다.</p></details></section>
''')

page('Transformer_Data_Flow_ELI5.html','데이터 흐름','''
<div class="small">문장 한 쌍을 따라가기 · 단어는 이해용 예시</div><h1>영어와 독일어 문장 한 쌍이<br>학습 한 번이 되기까지.</h1>
<section class="panel stage-steps">
<div><h2>① 문장 쌍 → 단어 조각</h2><div class="token-row"><span class="token">i</span><span class="token">love</span><span class="token">cats</span></div><p>spaCy가 문자열을 조각냅니다. Field는 소문자화와 시작·끝 기호 추가를 준비합니다.</p><p><a href="Transformer_Code_Walkthrough_ELI5.html#fn=Tokenizer.tokenize_en">Tokenizer.tokenize_en →</a></p></div>
<div><h2>② 단어 조각 → 번호표와 배치</h2><div class="box">훈련 자료로 vocab 만들기 → [B,S] 영어 · [B,T+1] 독일어</div><p>두 언어의 번호표는 별개입니다. 길이가 다른 문장은 패딩으로 맞춥니다. S와 T는 서로 달라도 됩니다. 실제 길이에는 특수 토큰도 포함됩니다.</p><p><a href="Transformer_Code_Walkthrough_ELI5.html#fn=DataLoader.build_vocab">build_vocab</a> · <a href="Transformer_Code_Walkthrough_ELI5.html#fn=DataLoader.make_iter">make_iter</a></p></div>
<div><h2>③ 정답을 한 칸 어긋나게 준비</h2><div class="contract"><div><b>Decoder에 주는 것</b><br>&lt;sos&gt; ich liebe katzen</div><div><b>맞혀야 하는 것</b><br>ich liebe katzen &lt;eos&gt;</div></div><p><code>trg[:, :-1]</code>을 입력으로, <code>trg[:, 1:]</code>을 정답으로 씁니다. 각 자리에서는 가림막 덕분에 자기 뒤의 정답을 볼 수 없습니다. 학습은 여러 자리의 점수를 한 번의 forward에서 계산합니다.</p></div>
<div><h2>④ Encoder → Decoder → 어휘 점수</h2><div class="diagram"><div class="box">영어 [B,S] → Encoder 메모 [B,S,512]</div><div class="arrow">↓ 독일어 앞부분과 함께 참고</div><div class="box after">Decoder → 점수 [B,T,V]</div></div><p>V는 독일어 어휘 수이며 자료의 번호표로 결정됩니다. 결과는 확률로 정규화하기 전의 logits입니다.</p></div>
<div><h2>⑤ 모양을 펼쳐 채점 → 수정</h2><div class="box">점수 [B×T,V] + 정답 [B×T] → Loss</div><p><code>loss.backward()</code> → 수정 방향 크기 제한 → <code>optimizer.step()</code> 순서로 모델을 갱신합니다. <a href="Transformer_Code_Walkthrough_ELI5.html#fn=train">train 함수 원문 →</a></p></div>
<div><h2>⑥ 한 바퀴 끝나면 검증하고 저장</h2><div class="box after">train → evaluate(valid_iter) → 더 좋은 가중치 저장</div><p>valid_loss가 가장 작으면 state_dict를 저장합니다. loss·BLEU 기록은 텍스트로 남기고 graph.py가 읽어 그래프로 보여줍니다.</p></div></section>
<section class="note"><b>현재 평가 코드의 범위</b><p>evaluate도 정답 앞부분을 제공하므로 자유 번역 평가와 다릅니다. 또한 get_bleu에 넘기는 자료 구조가 맞지 않아 단어 문자열을 문자 n-gram으로 처리합니다. 기록값을 표준 문장 BLEU로 해석하지 않도록 함수 해설에서 실제 동작을 함께 설명했습니다. 계산 코드는 수정하지 않았습니다.</p></section>
''')

def code_lines(source):
    return ''.join('<span class="line"><span class="num">'+str(i)+'</span>'+escape(line)+'</span>' for i,line in enumerate(source.splitlines(),1))

source_content='<div class="small">학습·모델 소스 원문 · 비어 있는 __init__.py와 교재 생성·검사 스크립트는 제외</div><h1>파일을 펼치면<br>코드 전체를 읽을 수 있어요.</h1><p>위 함수 해설의 줄 번호와 같은 원본 파일 기준입니다. import나 전역 코드도 포함합니다. 교재 생성 시점의 소스 스냅샷이며 HTML에서는 Python을 실행하지 않습니다.</p><nav class="flow">'+''.join('<a href="#file='+escape(file)+'">'+escape(file)+'</a>' for file in sources)+'</nav>'
for file,source in sources.items():
    source_content+='<section class="panel source-section" id="file='+escape(file)+'"><h2 class="break">'+escape(file)+'</h2><details><summary>파일 전체 원문 펼치기 · '+str(len(source.splitlines()))+'줄</summary><pre class="code full-code">'+code_lines(source)+'</pre></details></section>'
source_content+='''<script>function revealSource(){const id=decodeURIComponent(location.hash.slice(1));const section=document.getElementById(id);if(section){section.querySelector('details').open=true;requestAnimationFrame(()=>section.scrollIntoView({block:'start'}));}}window.addEventListener('hashchange',revealSource);revealSource();</script>'''
page('Transformer_AI_Code_Book.html','전체 코드 교재',source_content)

data=dict(lessons=lessons,blocks=blocks,definitionCount=definition_count,sourceFiles=list(sources))
(HERE/'book-data.js').write_text('window.BOOK = '+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+';\n',encoding='utf-8')
(HERE/'README.md').write_text('# Transformer ELI5 학습 교재\n\n`index.html`을 브라우저에서 열면 됩니다. 인터넷이나 Python 실행 없이 읽을 수 있습니다.\n\n원본 .py 소스가 바뀌면 `python docs/book/build_book.py`로 원문과 줄 번호를 다시 생성합니다. 새 함수나 클래스가 생기면 생성 스크립트의 설명도 함께 갱신해야 합니다. `python docs/book/check_book.py`는 소스 일치·설명 줄·로컬 링크를 검사합니다.\n',encoding='utf-8')
assert len({item['key'] for item in lessons})==len(lessons)
for item in lessons+blocks:
    assert item['source']==chunk(item['file'],item['start'],item['start']+len(item['source'].splitlines())-1)
    for step in item['steps']: assert item['start']<=step['line']<item['start']+len(item['source'].splitlines())
    for key in item['calls']: assert key in by_key,key
print(f'Built {len(PAGES)} pages · {definition_count} definitions + 4 execution sections · {len(blocks)} diagram blocks · {len(sources)} source files')
