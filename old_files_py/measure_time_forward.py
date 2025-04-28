"""        
import random
        import pandas as pd
        
        # ----------------------------------------------------------------
        # ELSE branch: synthetic data for performance measurement
        # ----------------------------------------------------------------
        max_lengths_list = [100, 300, 1000, 3000, 10000, 30000, 100000]
        results = []  # will store dicts of {max_length, model_time, optimizer_time}

        batch_size = 16
        torch.cuda.memory._record_memory_history(max_entries=100000)
        
        for max_len in max_lengths_list:
            logging.info(f"Testing max length  {max_len}")

            accumulation_steps = 1
            optim.zero_grad()
            
            # ------------------------------------------------------------
            # 1. Construct random input/output data with padding = -1
            #    shape = [batch_size, max_len, num_features]
            # ------------------------------------------------------------
            in_data_tensor = torch.full(
                (batch_size, max_len, 3),
                fill_value=-1.0,
                dtype=torch.float32,
                device=device
            )
            out_data_tensor = torch.full(
                (batch_size, max_len, 5),
                fill_value=-1.0,
                dtype=torch.float32,
                device=device
            )
            length_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)

            

            # For each event in the batch, choose random length in [0.9*max_len, max_len]
            for i in range(batch_size):
                min_len = int(0.9 * max_len)
                actual_len = random.randint(min_len, max_len)
                length_tensor[i] = actual_len

                # Fill the valid portion [0 : actual_len] with random floats
                # Typically you'd do something more domain-specific
                in_data_tensor[i, :actual_len, :] = torch.rand((actual_len, 3), device=device)
                out_data_tensor[i, :actual_len, :] = torch.rand((actual_len, 5), device=device)
            
            padding_mask = (in_data_tensor == -1.0).all(dim=-1)
            flex_padding_mask = generate_padding_mask(length_tensor)
            torch.cuda.reset_peak_memory_stats(device=device)
           


            # ------------------------------------------------------------
            # 2. Forward pass and measure model time
            # ------------------------------------------------------------


            #pred = model(in_data_tensor, padding_mask =padding_mask)

            with torch.amp.autocast('cuda'):  
                pred = model(in_data_tensor, f'train_{max_len}', flex_padding_mask, timer)
            


                # Create a mask based on actual lengths so we only compute loss on valid entries
                mask = torch.arange(max_len, device=device).expand(batch_size, max_len)
                mask = mask < length_tensor.unsqueeze(1)  # shape [B, S]

                # Flatten for the loss
                pred_valid = pred[mask, :]
                out_valid = out_data_tensor[mask, :]

                # Compute loss
                loss = loss_fn(pred_valid, out_valid)
            

            
            model_peak_mem = torch.cuda.max_memory_allocated(device=device)

            # ------------------------------------------------------------
            # 3. Backprop and measure optimizer time
            # ------------------------------------------------------------
            
            torch.cuda.reset_peak_memory_stats(device=device)
            if timer:
                timer.start("optimizer_time")

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            if timer:
                timer.stop()

            optimizer_peak_mem = torch.cuda.max_memory_allocated(device=device)

            epoch_stats = timer.get_stats(reset=True) 
            print(epoch_stats, flush=True)

            model.wandb_logger.log({"train/max_len": max_len})

            # Store in results
            results.append({
                "max_length": max_len,
                "encoder_time": epoch_stats['encoder'],
                "optimizer_time": epoch_stats['optimizer_time'],
                "model_peak_mem": model_peak_mem,         # in bytes
                "optimizer_peak_mem": optimizer_peak_mem  # in bytes
            })
            torch.cuda.empty_cache()

        # Dump memory snapshot history to a file and stop recording
        torch.cuda.memory._dump_snapshot("/projects/0/nisei0750/slava/data/cuda_profile_flex.pkl")
        torch.cuda.memory._record_memory_history(enabled=None)
        
        # ------------------------------------------------------------
        # 4. Convert results to DataFrame and save to CSV
        # ------------------------------------------------------------
        df_results = pd.DataFrame(results)
        csv_path = "/projects/0/nisei0750/slava/data/benchmark_results.csv"
        df_results.to_csv(csv_path, index=False)
        logging.info(f"Saved benchmark results to {csv_path}")

        assert False
"""