-- Nicolas Alt, 2014-09-04
-- Cheng Chi, 2023-07-27
-- Command-and-measure script
-- Tests showed about 30Hz rate
require "socket"
cmd.register(0xB0); -- Measure only
cmd.register(0xB1); -- Position PD

function hasbit(x, p)
  return x % (p + p) >= p       
end

function send_state()
    -- ==== Get measurements ====
    state = gripper.state();
    pos = mc.position();
    speed = mc.speed();
    force = mc.aforce();
    time = socket.gettime();

    if cmd.online() then
        -- Only the lowest byte of state is sent!
        cmd.send(id, etob(E_SUCCESS), state % 256, ntob(pos), ntob(speed), ntob(force), ntob(time));
    end
end

function process()
    id, payload = cmd.read();
    
    -- Position control
    if id == 0xB1 then
        -- get args
        cmd_pos = bton({payload[2],payload[3],payload[4],payload[5]});
        cmd_vel = bton({payload[6],payload[7],payload[8],payload[9]});
        cmd_kp = bton({payload[10],payload[11],payload[12],payload[13]});
        cmd_kd = bton({payload[14],payload[15],payload[16],payload[17]});
        cmd_travel_force_limit = bton({payload[18],payload[19],payload[20],payload[21]});
        cmd_blocked_force_limit = bton({payload[22],payload[23],payload[24],payload[25]});

        -- get state
        pos = mc.position();
        vel = mc.speed();
        
        -- pd controller
        e = cmd_pos - pos;
        de = cmd_vel - vel;
        act_vel = cmd_kp * e + cmd_kd * de;
        
        -- command
        mc.speed(act_vel);
        
        -- force limit
        if mc.blocked() then
            mc.force(cmd_blocked_force_limit);
        else
            mc.force(cmd_travel_force_limit);
        end
    end
    
    --t_start = socket.gettime();
    send_state();
    --print(socket.gettime() - t_start);
    
end

while true do
    if cmd.online() then
        -- process()
        if not pcall(process) then
            print("Error occured")
            sleep(100)
        end
    else
        sleep(100)
    end
end
