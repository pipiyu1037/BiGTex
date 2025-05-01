import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
from transformers import BertTokenizer, BertModel
from peft import get_peft_model, LoraConfig
from tqdm import tqdm

class BiGTexDGL(nn.Module):
    """
    DGL版本的BiGTex模型，结合图神经网络和BERT语言模型
    """
    def __init__(self, feature_dim, text_embedding_dim, num_classes, texts, embedding_dim=128, num_gcn_layers=2, Lora=True, soft=True):
        super(BiGTexDGL, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.texts = texts
        self.soft = soft
            
        for param in self.text_model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["attention.self.query", "attention.self.key", "attention.self.value"],
            lora_dropout=0.1,
            bias="none"
        )
        self.text_model = get_peft_model(self.text_model, lora_config)

        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, text_embedding_dim)
        )
        
        self.gcn_layers = nn.ModuleList([
            dglnn.SAGEConv(text_embedding_dim, text_embedding_dim, 'mean')
            for _ in range(num_gcn_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(text_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.cross_attention = nn.MultiheadAttention(text_embedding_dim, 4)

    def forward(self, graph, feature_vec, n_id):
        
        transformed_feature = self.feature_transform(feature_vec)

        text = [self.texts[i] for i in n_id.cpu().numpy()]

        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')

        device = graph.device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        input_embeddings = self.text_model.get_input_embeddings()(tokens['input_ids'])
        
        if self.soft == False:
            ## w\o soft prompt------------
            outputs = self.text_model(inputs_embeds=input_embeddings)
            hidden_states = outputs.last_hidden_state
            text_embedding = hidden_states[:, 0, :]
            ## w\o soft prompt------------

        graph_embedding = transformed_feature
        
        for gcn_layer in self.gcn_layers:
            if self.soft:
                ## w\ soft prompt
                # 图对文本进行注意力
                graph_embedding = graph_embedding.unsqueeze(1)

                modified_embeddings = torch.cat((graph_embedding, input_embeddings), dim=1)
                attention_mask = tokens['attention_mask']
                batch_size = attention_mask.shape[0]
                new_token_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([new_token_mask, attention_mask], dim=1)
                outputs = self.text_model(inputs_embeds=modified_embeddings, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
                text_embedding = hidden_states[:, 0, :]
                ## w\ soft prompt

            # 文本对图进行注意力
            graph_embedding_for_attention = graph_embedding.squeeze(1).unsqueeze(0)  # [1, batch_size, embedding_dim]
            text_embedding_for_attention = text_embedding.unsqueeze(0)  # [1, batch_size, embedding_dim]
            text_to_graph_attention, _ = self.cross_attention(
                graph_embedding_for_attention, 
                text_embedding_for_attention,
                text_embedding_for_attention
            )
            text_to_graph_attention = text_to_graph_attention.squeeze(0)  # [batch_size, embedding_dim]

            # 结合图→文本和文本→图的注意力
            combined_embedding = (text_embedding + text_to_graph_attention) / 2
            
            # 在DGL中更新图嵌入
            # 首先，将节点特征设置为combined_embedding
            # graph.ndata['h'] = combined_embedding
            # 然后，使用DGL的消息传递机制更新节点特征
            graph_embedding = gcn_layer(graph, combined_embedding)

        return self.classifier(graph_embedding), graph_embedding