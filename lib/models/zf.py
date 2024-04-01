
import torch.nn as nn

class SentenceProcessor:
    def __init__(self):
        """
        初始化句子处理器。

        Args:
        pred_layer: 用于计算概率值的预测层。
        """
        self.pred_layer1 = nn.Conv2d(512, 1, 1, 1)

    def process_data(self, sentence_mask, fused_h_t,batch_size,num_sentences):
        """
        处理数据并返回p值张量。

        Args:
        sentence_mask: 句子掩码张量。
        fused_h_t: 合并后的特征张量。

        Returns:
        p_values_tensor: 处理后的p值张量。
        """
        # sentence_mask = sentence_mask.view(sentence_mask.shape[0], 8, 1, 1, 1)
        # batch_size, num_sentences = sentence_mask.shape[:2]
        #
        # fused_h_t = fused_h_t.view(batch_size, 8, 512, 32, 32)
        p_values = []

        for batch in range(batch_size):
            p_values_batch = []
            prev_pp = None

            for sentence in range(num_sentences):
                if sentence_mask[batch, sentence, 0, 0, 0] == 1:
                    feature = fused_h_t[batch, sentence, :, :, :].view(1, 512, 32, 32)
                    if prev_pp is not None:
                        feature *= prev_pp

                    p = self.pred_layer(feature)
                    p = torch.sigmoid(p)
                    p_values_batch.append(p)

                    if prev_pp is not None:
                        pp = prev_pp - p
                    else:
                        pp = 1 - p
                    prev_pp = pp
                else:
                    p_values_batch.append(torch.zeros(1, 1, 32, 32).cuda())

            while len(p_values_batch) < num_sentences:
                p_values_batch.append(torch.zeros(1, 1, 32, 32).cuda())

            p_values.append(torch.cat(p_values_batch, 0))

        p_values_tensor = torch.stack(p_values, 0)
        return p_values_tensor