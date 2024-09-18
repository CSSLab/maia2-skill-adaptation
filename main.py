import chess.pgn
import chess
import pdb
from multiprocessing import Pool, cpu_count, Queue, Process
import torch
import tqdm
import argparse
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from tqdm.contrib.concurrent import process_map
import os
import pandas as pd
import time
from einops import rearrange


def process_chunks(cfg, pgn_path, pgn_chunks, elo_dict):

    # process_per_chunk((pgn_chunks[0][0], pgn_chunks[0][1], pgn_path, elo_dict, cfg))

    if cfg.verbose:
        results = process_map(
            process_per_chunk,
            [(start, end, pgn_path, elo_dict, cfg) for start, end in pgn_chunks],
            max_workers=len(pgn_chunks),
            chunksize=1,
        )
    else:
        with Pool(processes=len(pgn_chunks)) as pool:
            results = pool.map(
                process_per_chunk,
                [(start, end, pgn_path, elo_dict, cfg) for start, end in pgn_chunks],
            )

    ret = []
    count = 0
    list_of_dicts = []
    for result, game_count, frequency in results:
        ret.extend(result)
        count += game_count
        list_of_dicts.append(frequency)

    total_counts = {}

    for d in list_of_dicts:
        for key, value in d.items():
            total_counts[key] = total_counts.get(key, 0) + value

    print(total_counts, flush=True)

    return ret, count, len(pgn_chunks)


def process_per_game(game, white_elo, black_elo, white_win, cfg):

    ret = []

    board = game.board()
    moves = list(game.mainline_moves())

    for i, node in enumerate(game.mainline()):

        move = moves[i]

        if i >= cfg.first_n_moves:

            comment = node.comment
            clock_info = extract_clock_time(comment)

            if i % 2 == 0:
                board_input = board.fen()
                move_input = move.uci()
                elo_self = white_elo
                elo_oppo = black_elo
                active_win = white_win

            else:
                board_input = board.mirror().fen()
                move_input = mirror_move(move.uci())
                elo_self = black_elo
                elo_oppo = white_elo
                active_win = -white_win

            if clock_info > cfg.clock_threshold:
                ret.append((board_input, move_input, elo_self, elo_oppo, active_win))

        board.push(move)
        if i == cfg.max_ply:
            break

    return ret


def game_filter(game):

    white_elo = game.headers.get("WhiteElo", "?")
    black_elo = game.headers.get("BlackElo", "?")
    time_control = game.headers.get("TimeControl", "?")
    result = game.headers.get("Result", "?")
    event = game.headers.get("Event", "?")

    if (
        white_elo == "?"
        or black_elo == "?"
        or time_control == "?"
        or result == "?"
        or event == "?"
    ):
        return

    if "Rated" not in event:
        return

    if "Rapid" not in event:
        return

    for _, node in enumerate(game.mainline()):
        if "clk" not in node.comment:
            return

    white_elo = int(white_elo)
    black_elo = int(black_elo)

    if result == "1-0":
        white_win = 1
    elif result == "0-1":
        white_win = -1
    elif result == "1/2-1/2":
        white_win = 0
    else:
        return

    return game, white_elo, black_elo, white_win


def process_per_chunk(args):

    start_pos, end_pos, pgn_path, elo_dict, cfg = args

    ret = []
    game_count = 0

    frequency = {}

    with open(pgn_path, "r", encoding="utf-8") as pgn_file:

        pgn_file.seek(start_pos)

        while pgn_file.tell() < end_pos:

            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            filtered_game = game_filter(game)
            if filtered_game:
                game, white_elo, black_elo, white_win = filtered_game
                white_elo = map_to_category(white_elo, elo_dict)
                black_elo = map_to_category(black_elo, elo_dict)

                if white_elo < black_elo:
                    range_1, range_2 = black_elo, white_elo
                else:
                    range_1, range_2 = white_elo, black_elo

                freq = frequency.get((range_1, range_2), 0)
                if freq >= cfg.max_games_per_elo_range:
                    continue

                ret_per_game = process_per_game(
                    game, white_elo, black_elo, white_win, cfg
                )
                ret.extend(ret_per_game)
                if len(ret_per_game):

                    if (range_1, range_2) in frequency:
                        frequency[(range_1, range_2)] += 1
                    else:
                        frequency[(range_1, range_2)] = 1

                    game_count += 1

    return ret, game_count, frequency


class MAIA1Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, all_moves_dict, elo_dict, cfg):
        
        self.all_moves_dict = all_moves_dict
        self.cfg = cfg
        self.data = data.values.tolist()
        self.elo_dict = elo_dict
    
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        fen, move, elo_self, elo_oppo, white_active = self.data[idx]

        if white_active:
            board = chess.Board(fen)
        else:
            board = chess.Board(fen).mirror()
            move = mirror_move(move)
            
        board_input = board_to_tensor(board)
        move_input = self.all_moves_dict[move]
        
        elo_self = map_to_category(elo_self, self.elo_dict)
        elo_oppo = map_to_category(elo_oppo, self.elo_dict)
        
        legal_moves, side_info = get_side_info(board, move, self.all_moves_dict)
        
        return board_input, move_input, elo_self, elo_oppo, legal_moves, side_info


class MAIA2Dataset(torch.utils.data.Dataset):
    
    
    def __init__(self, data, all_moves_dict, cfg):
        
        self.all_moves_dict = all_moves_dict
        self.data = data
        self.cfg = cfg
    
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        board_input, move_uci, elo_self, elo_oppo, active_win = self.data[idx]
        
        board = chess.Board(board_input)
        board_input = board_to_tensor(board)
        
        legal_moves, side_info = get_side_info(board, move_uci, self.all_moves_dict)
        
        move_input = self.all_moves_dict[move_uci]
        
        return board_input, move_input, elo_self, elo_oppo, legal_moves, side_info, active_win, board_input

class BasicBlock(torch.nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        mid_planes = planes
        
        self.conv1 = torch.nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(mid_planes)
        self.conv2 = torch.nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = F.relu(out)

        return out


class ChessResNet(torch.nn.Module):
    
    def __init__(self, block, cfg):
        super(ChessResNet, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(cfg.input_channels, cfg.dim_cnn, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(cfg.dim_cnn)
        self.layers = self._make_layer(block, cfg.dim_cnn, cfg.num_blocks_cnn)
        self.conv_last = torch.nn.Conv2d(cfg.dim_cnn, cfg.vit_length, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_last = torch.nn.BatchNorm2d(cfg.vit_length)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, planes, stride))
        
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.conv_last(out)
        out = self.bn_last(out)
        
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class EloAwareAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., elo_dim=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.elo_query = nn.Linear(elo_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, elo_emb):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        elo_effect = self.elo_query(elo_emb).view(x.size(0), self.heads, 1, -1)
        q = q + elo_effect

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., elo_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.elo_layers = nn.ModuleList([])
        for _ in range(depth):
            self.elo_layers.append(nn.ModuleList([
                EloAwareAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, elo_dim = elo_dim),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, elo_emb):
        for attn, ff in self.elo_layers:
            x = attn(x, elo_emb) + x
            x = ff(x) + x

        return self.norm(x)


class MAIA2Model(torch.nn.Module):
    
    def __init__(self, output_dim, elo_dict, cfg):
        super(MAIA2Model, self).__init__()
        
        self.cfg = cfg
        self.chess_cnn = ChessResNet(BasicBlock, cfg)
        
        heads = 16
        dim_head = 64
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(8 * 8, cfg.dim_vit),
            nn.LayerNorm(cfg.dim_vit),
        )
        self.transformer = Transformer(cfg.dim_vit, cfg.num_blocks_vit, heads, dim_head, mlp_dim=cfg.dim_vit, dropout = 0.1, elo_dim = cfg.elo_dim * 2)
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.vit_length, cfg.dim_vit))
        
        self.fc_1 = nn.Linear(cfg.dim_vit, output_dim)
        # self.fc_1_1 = nn.Linear(cfg.dim_vit, cfg.dim_vit)
        self.fc_2 = nn.Linear(cfg.dim_vit, output_dim + 6 + 6 + 1 + 64 + 64)
        # self.fc_2_1 = nn.Linear(cfg.dim_vit, cfg.dim_vit)
        self.fc_3 = nn.Linear(128, 1)
        self.fc_3_1 = nn.Linear(cfg.dim_vit, 128)
        
        self.elo_embedding = torch.nn.Embedding(len(elo_dict), cfg.elo_dim)
        
        self.dropout = nn.Dropout(p=0.1)
        self.last_ln = nn.LayerNorm(cfg.dim_vit)


    def forward(self, boards, elos_self, elos_oppo):
        
        batch_size = boards.size(0)
        boards = boards.view(batch_size, self.cfg.input_channels, 8, 8)
        embs = self.chess_cnn(boards)
        embs = embs.view(batch_size, embs.size(1), 8 * 8)
        x = self.to_patch_embedding(embs)
        x += self.pos_embedding
        x = self.dropout(x)
        
        elos_emb_self = self.elo_embedding(elos_self)
        elos_emb_oppo = self.elo_embedding(elos_oppo)
        elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)
        x = self.transformer(x, elos_emb).mean(dim=1)
        
        x = self.last_ln(x)

        logits_maia = self.fc_1(x)
        logits_side_info = self.fc_2(x)
        logits_value = self.fc_3(self.dropout(torch.relu(self.fc_3_1(x)))).squeeze(dim=-1)
        
        return logits_maia, logits_side_info, logits_value


def read_monthly_data_path(cfg):
    
    print('Training Data:', flush=True)
    pgn_paths = []
    
    for year in range(cfg.start_year, cfg.end_year + 1):
        start_month = cfg.start_month if year == cfg.start_year else 1
        end_month = cfg.end_month if year == cfg.end_year else 12

        for month in range(start_month, end_month + 1):
            formatted_month = f"{month:02d}"
            pgn_path = cfg.data_root + f"/lichess_db_standard_rated_{year}-{formatted_month}.pgn"
            print(pgn_path, flush=True)
            pgn_paths.append(pgn_path)
            
    return pgn_paths


def evaluate(model, dataloader):
    
    counter = 0
    correct_move = 0
    
    model.eval()
    with torch.no_grad():
        
        for boards, labels, elos_self, elos_oppo, legal_moves, side_info in dataloader:
            
            boards = boards.cuda()
            labels = labels.cuda()
            elos_self = elos_self.cuda()
            elos_oppo = elos_oppo.cuda()
            legal_moves = legal_moves.cuda()

            logits_maia, logits_side_info, logits_value = model(boards, elos_self, elos_oppo)
            logits_maia_legal = logits_maia * legal_moves
            preds = logits_maia_legal.argmax(dim=-1)
            correct_move += (preds == labels).sum().item()
            
            counter += len(labels)

    return correct_move, counter


def evaluate_MAIA1_data(model, all_moves_dict, elo_dict, cfg, tiny=False):
    
    elo_list = range(1000, 2600, 100)

    for i in elo_list:
        start = i
        end = i + 100
        file_path = f"../data/test/KDDTest_{start}-{end}.csv"
        data = pd.read_csv(file_path)
        data = data[data.type == 'Rapid'][['board', 'move', 'active_elo', 'opponent_elo', 'white_active']]
        dataset = MAIA1Dataset(data, all_moves_dict, elo_dict, cfg)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=cfg.batch_size, 
                                                shuffle=False, 
                                                drop_last=False,
                                                num_workers=cfg.num_workers)
        if cfg.verbose:
            dataloader = tqdm.tqdm(dataloader)
        print(f'Testing Elo Range {start}-{end} with MAIA 1 data:', flush=True)
        correct_move, counter = evaluate(model, dataloader)
        print(f'Accuracy Move Prediction: {round(correct_move / counter, 4)}', flush=True)
        if tiny:
            break


def train_chunks(cfg, data, model, optimizer, all_moves_dict, criterion_maia, criterion_side_info, criterion_value):
    
    dataset_train = MAIA2Dataset(data, all_moves_dict, cfg)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, 
                                                    batch_size=cfg.batch_size, 
                                                    shuffle=True, 
                                                    drop_last=False,
                                                    num_workers=cfg.num_workers)
    if cfg.verbose:
        dataloader_train = tqdm.tqdm(dataloader_train)
    
    avg_loss = 0
    avg_loss_maia = 0
    avg_loss_side_info = 0
    avg_loss_value = 0
    step = 0
    for boards, labels, elos_self, elos_oppo, legal_moves, side_info, wdl in dataloader_train:

        model.train()
        boards = boards.cuda()
        labels = labels.cuda()
        elos_self = elos_self.cuda()
        elos_oppo = elos_oppo.cuda()
        side_info = side_info.cuda()
        wdl = wdl.float().cuda()
        
        logits_maia, logits_side_info, logits_value = model(boards, elos_self, elos_oppo)
        
        loss = 0
        loss_maia = criterion_maia(logits_maia, labels)
        loss += loss_maia
        
        if cfg.side_info:
        
            loss_side_info = criterion_side_info(logits_side_info, side_info) * cfg.side_info_coefficient
            loss += loss_side_info
        
        if cfg.value:
            loss_value = criterion_value(logits_value, wdl) * cfg.value_coefficient
            loss += loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        avg_loss_maia += loss_maia.item()
        if cfg.side_info:
            avg_loss_side_info += loss_side_info.item()
        if cfg.value:
            avg_loss_value += loss_value.item()
        step += 1
    
    return round(avg_loss / step, 3), round(avg_loss_maia / step, 3), round(avg_loss_side_info / step, 3), round(avg_loss_value / step, 3)


def preprocess_thread(queue, cfg, pgn_path, pgn_chunks_sublist, elo_dict):
    
    data, game_count, chunk_count = process_chunks(cfg, pgn_path, pgn_chunks_sublist, elo_dict)
    queue.put([data, game_count, chunk_count])
    del data


def worker_wrapper(semaphore, *args, **kwargs):
    with semaphore:
        preprocess_thread(*args, **kwargs)


def parse_args(args=None):

    parser = argparse.ArgumentParser()

    # Supporting Arguments
    # parser.add_argument('--data_root', default='../data', type=str)
    parser.add_argument('--data_root', default='/datadrive2/lichess_data', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--max_ply', default=300, type=int)
    parser.add_argument('--clock_threshold', default=30, type=int)
    parser.add_argument('--chunk_size', default=20000, type=str)
    parser.add_argument('--start_year', default=2021, type=int)
    parser.add_argument('--start_month', default=7, type=int)
    parser.add_argument('--end_year', default=2023, type=int)
    parser.add_argument('--end_month', default=11, type=int)
    parser.add_argument('--from_checkpoint', default=True, type=bool)
    parser.add_argument('--checkpoint_year', default=2021, type=int)
    parser.add_argument('--checkpoint_month', default=6, type=int)
    parser.add_argument('--test_year', default=2019, type=int)
    parser.add_argument('--test_month', default=10, type=int)
    parser.add_argument('--num_cpu_left', default=16, type=int)
    parser.add_argument('--queue_length', default=2, type=int)
    parser.add_argument('--max_games_per_elo_range', default=20, type=int)

    # Tunable Arguments
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8192, type=int)
    parser.add_argument('--first_n_moves', default=10, type=int)
    parser.add_argument('--last_n_moves', default=10, type=int)
    parser.add_argument('--dim_cnn', default=256, type=int)
    parser.add_argument('--dim_vit', default=1024, type=int)
    parser.add_argument('--num_blocks_cnn', default=5, type=int)
    parser.add_argument('--num_blocks_vit', default=2, type=int)
    parser.add_argument('--input_channels', default=18, type=int)
    parser.add_argument('--vit_length', default=8, type=int)
    parser.add_argument('--elo_dim', default=128, type=int)
    parser.add_argument('--side_info', default=True, type=bool)
    parser.add_argument('--side_info_coefficient', default=1, type=float)
    parser.add_argument('--value', default=True, type=bool)
    parser.add_argument('--value_coefficient', default=1, type=float)

    return parser.parse_args(args)


if __name__ == '__main__':
    
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    num_processes = cpu_count() - cfg.num_cpu_left
    
    save_root = f'../tmp/{cfg.lr}_{cfg.batch_size}_{cfg.wd}_MAIA2/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    model = MAIA2Model(len(all_moves), elo_dict, cfg)

    print(model, flush=True)
    model = model.cuda()
    model = nn.DataParallel(model)
    criterion_maia = nn.CrossEntropyLoss()
    criterion_side_info = nn.BCEWithLogitsLoss()
    criterion_value = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    N_params = count_parameters(model)
    print(f'Trainable Parameters: {N_params}', flush=True)
    
    accumulated_samples = 0
    accumulated_games = 0
    
    if cfg.from_checkpoint:
        formatted_month = f"{cfg.checkpoint_month:02d}"
        checkpoint = torch.load(save_root + f'epoch_1_{cfg.checkpoint_year}-{formatted_month}.pgn.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        accumulated_samples = checkpoint['accumulated_samples']
        accumulated_games = checkpoint['accumulated_games']

    for epoch in range(cfg.max_epochs):
        
        print(f'Epoch {epoch + 1}', flush=True)
        pgn_paths = read_monthly_data_path(cfg)
        
        num_file = 0
        for pgn_path in pgn_paths:
            
            start_time = time.time()
            decompress_zst(pgn_path + '.zst', pgn_path)
            print(f'Decompressing {pgn_path} took {readable_time(time.time() - start_time)}', flush=True)

            pgn_chunks = read_or_create_chunks(pgn_path, cfg)
            print(f'Training {pgn_path} with {len(pgn_chunks)} chunks.', flush=True)
            
            queue = Queue(maxsize=cfg.queue_length)
            
            pgn_chunks_sublists = []
            for i in range(0, len(pgn_chunks), num_processes):
                pgn_chunks_sublists.append(pgn_chunks[i:i + num_processes])
            
            pgn_chunks_sublist = pgn_chunks_sublists[0]
            # This line is for debugging only
            # process_chunks(cfg, pgn_path, pgn_chunks_sublist, elo_dict)
            worker = Process(target=preprocess_thread, args=(queue, cfg, pgn_path, pgn_chunks_sublist, elo_dict))
            worker.start()
            
            num_chunk = 0
            offset = 0
            while True:
                if not queue.empty():
                    if offset + 1 < len(pgn_chunks_sublists):
                        pgn_chunks_sublist = pgn_chunks_sublists[offset + 1]
                        worker = Process(target=preprocess_thread, args=(queue, cfg, pgn_path, pgn_chunks_sublist, elo_dict))
                        worker.start()
                        offset += 1
                    data, game_count, chunk_count = queue.get()
                    loss, loss_maia, loss_side_info, loss_value = train_chunks(cfg, data, model, optimizer, all_moves_dict, criterion_maia, criterion_side_info, criterion_value)
                    num_chunk += chunk_count
                    accumulated_samples += len(data)
                    accumulated_games += game_count
                    print(f'[{num_chunk}/{len(pgn_chunks)}]', flush=True)
                    print(f'[# Positions]: {readable_num(accumulated_samples)}', flush=True)
                    print(f'[# Games]: {readable_num(accumulated_games)}', flush=True)
                    print(f'[# Loss]: {loss} | [# Loss MAIA]: {loss_maia} | [# Loss Side Info]: {loss_side_info} | [# Loss Value]: {loss_value}', flush=True)
                    if num_chunk == len(pgn_chunks):
                        break

            num_file += 1
            print(f'### ({num_file} / {len(pgn_paths)}) Took {readable_time(time.time() - start_time)} to train {pgn_path} with {len(pgn_chunks)} chunks.', flush=True)
            os.remove(pgn_path)
            
            evaluate_MAIA1_data(model, all_moves_dict, elo_dict, cfg, tiny=False)
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accumulated_samples': accumulated_samples,
                        'accumulated_games': accumulated_games}, f'{save_root}epoch_{epoch + 1}_{pgn_path[-11:]}.pt')
