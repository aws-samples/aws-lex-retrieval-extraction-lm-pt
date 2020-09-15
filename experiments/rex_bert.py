import torch
import torch.nn as nn

from transformers import *

class MIPRetriever(nn.Module):
    '''

    A potentially useful Max Inner Product retriever module (bi-encoder)
    Cross encoders are generally better, but bi-encoders are faster.

    '''
    def __init__(self):
        super().__init__()

    def forward(self, query, context, batch_size, num_ctx):
        ret_logits = torch.bmm(query, context.transpose(1, 2)).squeeze(1)
        _, ret_pred = ret_logits.max(dim=1)
        return ret_logits, ret_pred

class RexBert(DistilBertPreTrainedModel):
    '''

    The RexBert class implements the retrieval-extraction BERT for pretraining
    There are two subtasks of the model:
    1. Retrieval for a set of sampled contexts, generate cross entropy loss with given label
    2. Extract from the ground truth contexts

    '''

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel.from_pretrained(
            'distilbert-base-cased',
            config=config
        )
        self.retriever = MIPRetriever()
        self.ret_outputs = nn.Linear(config.hidden_size, 1)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.embed_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.embed_ctx = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

        self.distilbert = nn.DataParallel(self.distilbert)
        self.qa_outputs = nn.DataParallel(self.qa_outputs)
        self.ret_outputs = nn.DataParallel(self.ret_outputs)

    def forward_ret_ext(self, rex_batch, ret=True, ext=True):
        '''

        A 2-step, MIP based retriever before extractive training.
        Uses the retrieve() function.

        '''
        ret_loss = 0
        ext_loss = 0
        if ret:
            ret_loss, ret_pred = self.retrieve(rex_batch)
            ret_loss = ret_loss.mean()
            return ret_loss
        elif ext:
            with torch.no_grad():
                _, ret_pred = self.retrieve(rex_batch)
            ext_loss = self.extract(rex_batch, ret_pred)
            ext_loss = ext_loss.mean()
            return ext_loss
        return ret_loss, ext_loss

    def retrieve(self, rex_batch):
        '''

        An MIP-based retriever
        
        Calculates the sequence embeddings of queries and contexts, and
            Inner product matrix among them.

        '''
        ques_ids = rex_batch.ques_ids
        ques_len = ques_ids.size(1)
        ques_attention_mask = rex_batch.ques_attention_mask
        ctx_ids = torch.cat(rex_batch.ctx_ids, 0)
        ctx_len = ctx_ids.size(1)
        ctx_attention_mask = torch.cat(rex_batch.ctx_attention_mask, 0)
        labels = rex_batch.labels

        ques_output = self.distilbert(
            input_ids=ques_ids,
            attention_mask=ques_attention_mask
        )
        ques_emb, _ = ques_output[0].split([1, ques_len - 1], dim=1)
        ques_emb = self.embed_query(ques_emb)
        batch_size = rex_batch.batch_size

        ctx_output = self.distilbert(input_ids=ctx_ids, attention_mask=ctx_attention_mask)
        ctx_emb, _ = ctx_output[0].split([1, ctx_len - 1], dim=1)
        ctx_emb = self.embed_ctx(ctx_emb)
        ret_logits, ret_pred = self.retriever(ques_emb, ctx_emb, batch_size, rex_batch.num_ctx)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(ret_logits, labels)
        return loss, ret_pred

    def extract(self, rex_batch, ret_pred):
        '''

        The Extract wrapper for the extract_forward function.

        '''
        ext_examples = rex_batch.concat_ques_ctx(ret_pred)
        outputs, _ = self.extract_forward(**ext_examples)
        loss = outputs[0]
        return loss

    def forward(self, rex_batch, ret=None, ext=None):
        '''

        The forward function for cross-encoder based retriever + extraction
        Returns extraction loss and retrieval accuracies

        '''
        inputs = rex_batch.data
        outputs, ret_acc = self.extract_forward(**inputs, ret=ret, ext=ext)
        loss = outputs[:2]
        return loss, ret_acc

    def extract_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        label = None,
        num_ctx = None,
        batch_size = None,
        idx_base = None,
        ret = None,
        ext = None,
        rex_training = True
    ):
        '''

        Calculates the extraction loss and retrieval accuracies.
        The extraction is based on start and end position index predictions.

        '''

        ret_acc = 0

        # Consider adding token type ids if using BERT
        # If DistilBERT, no token type id should be used
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        seq_len = sequence_output.size(1)

        if rex_training:
            cls_emb, _ = sequence_output.split([1, seq_len - 1], dim=1)
            cls_emb = cls_emb.squeeze(1)
            ret_logits = self.ret_outputs(cls_emb).squeeze(-1).view(batch_size, num_ctx)

            _, ret_pred = ret_logits.max(1)
            if label is not None:
                ret_acc = (ret_pred == label).float().sum() / batch_size

            if ret:
                ret_loss_fct = nn.CrossEntropyLoss()
                ret_loss = ret_loss_fct(ret_logits, label)
                # return [loss,], ret_acc

            ret_target = label + idx_base

            sequence_output = sequence_output.index_select(0, ret_target)
            start_positions = start_positions.index_select(0, ret_target)
            end_positions = end_positions.index_select(0, ret_target)

        # sequence_output = self.dropout_ext(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if ret:
                outputs = (total_loss, ret_loss) + outputs
            else:
                outputs = (total_loss, ) + outputs

        return outputs, ret_acc
