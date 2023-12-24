

def shard_text_loader(args, shard_id):
    import os
    import tarfile

    # huxu: replace with your text iterator.
    shard_group = shard_dir % 100
    wds_fn = f"{args.shard_dir}/{shard_group}/{shard_id}.tar"
    if os.path.exists(wds_fn):
        print(f"missing {wds_fn}")
        return

    json_uuid, img_uuid = None, None
    with tarfile.open(wds_fn) as tar:
        members = tar.getmembers()
        for member in members:
            if member.name.endswith(".jpeg"):
                img_uuid = member.name[:-len(".jpeg")]
                jpg_start_offset = member.offset_data
                jpg_end_offset = member.offset_data + member.size
            elif member.name.endswith(".json"):
                json_uuid = member.name[:-len(".json")]
                json_start_offset = member.offset_data
                json_end_offset = member.offset_data + member.size 
                with tar.extractfile(member) as f:
                    text_json = json.load(f)
            else:
                raise ValueError(f"unknown file ext {member.name} in {wds_fn}")

            if json_uuid is None or img_uuid is None:
                continue

            assert json_uuid == img_uuid  # jpeg / json has to be placed alternatively in tar.

            # scan text_json and build index
            for offset, (text_key, text) in enumerate(text_json["texts"]):
                yield text, ((jpg_start_offset, jpg_end_offset), (json_start_offset, json_end_offset, offset))


def main(args, step_str):
    if step_str == "substr_indexing":
        from metaclip.indexing.substr_indexing import build_shards_index
    
        with open(args.metadata) as f:
            metadata = json.load(f)

        build_shards_index(args.index_dir, metadata, lambda shard_id: shard_text_loader(args, shard_id), args.start_shard, args.end_shard)
    elif step_str == "entry_count":
        from metaclip.indexing.entry_count import entry_count
        
        entry_count(args)
    elif step_str == "balance_sampling":
        from metaclip.indexing.balance_sampling import build_subset_index

        build_subset_index(config)
    else:
        raise ValueError(f"unknown step in pipeline {step_str}")



if __name__ == '__main__':
    # python metaclip/pipeline.py metaclip_400m substr_indexing
    # python metaclip/pipeline.py metaclip_400m entry_count
    # python metaclip/pipeline.py metaclip_400m balance_sampling

    import sys

    sys.path.append("./")
    from configs import search_config
    config = search_config(sys.argv[1])

    main(config, sys.argv[2])
