#!/usr/bin/env python
# coding: utf-8

# Key to the definition of differential privacy is the ability to ask the question "When querying a database, if I removed someone from the database, would the output of the query be any different?". Thus, in oder to check this, we must construct what we term "parallel databases" which are simply databases with one entry removed.


import torch


def get_parallel_db(db, remove_index):
    return  torch.cat((db[0:remove_index], db[remove_index+1:]))


def get_parallel_dbs(db):
    parallel_dbs = list()
    for i in range(len(db)):
        pdb = get_parallel_db(db, i)
        parallel_dbs.append(pdb)
    
    return parallel_dbs

def create_db_and_parallels(num_entries):
    db = torch.rand(num_entries) > 0.5
    db = db.type(torch.uint8)
    pdbs = get_parallel_dbs(db)
    
    return db, pdbs