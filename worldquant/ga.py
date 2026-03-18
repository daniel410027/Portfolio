import numpy as np
import typing
import tqdm
import os
import time

# import alpha as alpha_lib  # 註解掉公司機密模塊
# import alpha_utils as alpha_utils_lib  # 註解掉公司機密模塊

import json
import pathlib
from typing import List, Dict

# import config  # 註解掉公司機密模塊

CONFIG = {
    'population': 100,
    'selection_count' : 60,
    'mutation_rate': 0.1,
    'iteration': 15
}





SPACE = {

}

SETTINGS_TEMPLATE = {
    "instrumentType": "EQUITY",
    "region": "ASI",
    "universe": "",
    "delay": 1,
    "decay": 0,
    "neutralization": "",
    "truncation": 0.08,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "maxTrade": "OFF",
    "language": "FASTEXPR",
    "visualization": False, 
    "testPeriod": "P2Y"  
}
class GeneticAlgo:
    def __init__(self, name: str, templates: typing.List[str], space: typing.Dict[str, typing.List[str]], settings_template: typing.Dict[str, str], config: typing.Dict = CONFIG):
        self.name = name
        self.templates = templates
        self.space = space
        self.settings_template = settings_template
        self.config = config

        self.gene_database: typing.Dict[str, typing.Dict[str, str]] = {}
        self.generation_database: typing.List[typing.Dict[str, typing.Any]] = []  # 註解掉 alpha_lib.Alpha

    @staticmethod
    def generate_alpha_name(prefix: str, iteration: int, ind: int):
        return f'{prefix}_{iteration}_{ind}'

    def generate_settings(self, genes: typing.Dict[str, str]):
        settings = self.settings_template.copy()
        # Randomly select decay, truncation, and neutralization from predefined spaces
        settings['decay'] = int(np.random.choice(self.space['decay']))
        settings['truncation'] = np.random.choice(
            self.space['truncation'])
        settings['neutralization'] = str(
            np.random.choice(self.space['neutralization']))

        # Apply genes to settings
        for gene_name, gene_val in genes.items():
            if gene_name in settings:
                settings[gene_name] = gene_val
        for key, value in settings.items():
            if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
                placeholder = value[1:-1]
                if placeholder in genes:
                    settings[key] = genes[placeholder]
        return settings

    def collect_completed_alphas(self) -> Dict[str, typing.Any]:  # 註解掉 alpha_lib.Alpha
        """從 complete_notify.json 中獲取已完成的 Alphas"""
        # notify_file_path = pathlib.Path(
        #     config.COMPLETE_NOTIFY_FILE)  # 使用config中的路径
        # if not notify_file_path.exists():
        #     return {}

        # with open(notify_file_path, 'r') as f:
        #     complete_names = json.load(f)

        # # 刪除通知文件
        # notify_file_path.unlink()

        # # 讀取已完成的 Alpha
        # completed_alphas = {}
        # for name in complete_names:
        #     completed_alphas[name] = alpha_lib.Alpha.read_from_db(
        #         os.path.join(config.COMPLETE_FOLDER, name))  # 使用config中的路径

        # return completed_alphas
        return {}  # 註解掉會跳bug的行

    def generate_initial_population(self):
        alphas: typing.Dict[str, typing.Any] = {}  # 註解掉 alpha_lib.Alpha
        for i in range(self.config['population']):
            name = self.generate_alpha_name(self.name, 0, i)
            template = np.random.choice(self.templates)
            expr = template
            genes = {}
            for gene_name, gene_vals in self.space.items():
                if gene_name in ['decay', 'neutralization', 'truncation']:
                    continue
                genes[gene_name] = np.random.choice(gene_vals)
                expr = expr.replace(gene_name, genes[gene_name])
            settings = self.generate_settings(genes)
            # alphas[name] = alpha_lib.Alpha(
            #     name,
            #     payload={
            #         "type": "REGULAR",
            #         "settings": settings,
            #         "regular": expr, 
            #     }
            # )
            # alphas[name].write_to_db()
            alphas[name] = {"name": name, "settings": settings, "expr": expr}  # 註解掉會跳bug的行
            self.gene_database[name] = genes
        # finish_alphas = self.collect_alphas(alphas)
        finish_alphas = alphas  # 註解掉會跳bug的行
        self.generation_database.append(finish_alphas)

    # def collect_alphas(self, alphas: typing.Dict[str, alpha_lib.Alpha]) -> typing.Dict[str, alpha_lib.Alpha]:
    #     uncollected_names = list(alphas.keys())
    #     ret_alphas = {}
    #     with tqdm.tqdm(len(uncollected_names)) as pbar:
    #         while len(uncollected_names) > 0:
    #             for name, alpha in alphas.items():
    #                 if name not in uncollected_names:
    #                     continue
    #                 # if os.path.isfile(os.path.join(config.COMPLETE_FOLDER, alpha.filename)):
    #                 #     ret_alphas[name] = alpha_lib.Alpha.read_from_db(
    #                 #         os.path.join(config.COMPLETE_FOLDER, alpha.filename))
    #                 #     uncollected_names.remove(name)
    #                 #     pbar.update(1)
    #                 time.sleep(1)
                
    #     return ret_alphas

    def selection(self, alphas: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:  # 註解掉 alpha_lib.Alpha
        names = np.array(list(alphas.keys()))
        # fitnesses = np.array([(alphas[name].result['is']['fitness'] or 0) * 2 + (alphas[name].result['is']['sharpe'] or 0) * 0
        #                       + (alphas[name].result['is']['margin'] or 0) * 0 for name in names])
        fitnesses = np.random.rand(len(names))  # 註解掉會跳bug的行，隨機生成適應度

        # 找到 fitnesses 的排序
        sorted_indices = np.argsort(fitnesses)[::-1]  # 降序排序
        top_count = self.config['selection_count']  # 使用 selection_count

        # 選擇適應度最高的前 selection_count 個個體
        selected_indices = sorted_indices[:top_count]

        # 使用選中的索引返回相應的 alphas
        return {names[i]: alphas[names[i]] for i in selected_indices}

    def crossover_mutation(self, gene_1: typing.Dict, gene_2: typing.Dict) -> typing.Dict:
        child_gene = {}
        for gene_name in gene_1.keys():
            child_gene[gene_name] = np.random.choice(
                [gene_1[gene_name], gene_2[gene_name]])
            if np.random.rand() < self.config['mutation_rate']:
                child_gene[gene_name] = np.random.choice(
                    self.space[gene_name])
        return child_gene

    def main(self):
        self.generate_initial_population()
        for generation_ind in range(1, self.config['iteration']+1):
            # 收集已完成的 Alphas
            completed_alphas = self.generation_database[-1]
            parents = self.selection(completed_alphas)  # 使用完成的 Alphas 進行選擇

            alphas: typing.Dict[str, typing.Any] = {}  # 註解掉 alpha_lib.Alpha
            for child_ind in range(self.config['population']):
                child_name = self.generate_alpha_name(
                    self.name, generation_ind, child_ind)
                parents_names = list(
                    np.random.choice(list(parents.keys()), 2))
                child_gene = self.crossover_mutation(
                    self.gene_database[parents_names[0]], self.gene_database[parents_names[1]])
                self.gene_database[child_name] = child_gene
                template = np.random.choice(self.templates)
                expr = template
                for gene_name, gene_val in self.gene_database[child_name].items():
                    expr = expr.replace(gene_name, gene_val)
                settings = self.generate_settings(
                    self.gene_database[child_name])
                # alphas[child_name] = alpha_lib.Alpha(
                #     child_name,
                #     payload={
                #         "type": "REGULAR",
                #         "settings": settings,
                #         "regular": expr, 
                #     }
                # )
                # alphas[child_name].write_to_db()
                alphas[child_name] = {"name": child_name, "settings": settings, "expr": expr}  # 註解掉會跳bug的行
            # 在 pending 文件夾中寫入新的 Alphas
            # for alpha in alphas.values():
            #     alpha.update_status(config.PENDING_FOLDER)
            # finish_alphas = self.collect_alphas(alphas)
            finish_alphas = alphas  # 註解掉會跳bug的行
            self.generation_database.append(finish_alphas)

if __name__ == '__main__':
    # templates = [TEMPLATE_VEC_AB]  # 註解掉公司機密模板
    templates = ["<TEMPLATE_VEC_AB>"]  # 註解掉會跳bug的行
    ga = GeneticAlgo(
        name='20250803_test_ga_022',
        templates=templates,
        space=SPACE,
        settings_template=SETTINGS_TEMPLATE,
        config=CONFIG
    )
    ga.main()

