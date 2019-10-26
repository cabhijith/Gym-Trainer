import aiohttp
import asyncio
import uvicorn
from fastai import * 
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles


export_file_url = 'https://www.googleapis.com/drive/v3/files/1-IUnAKJhmOm1QKwQJCbDI8aOxtW8V7kb?alt=media&key=AIzaSyByRUuaBnB4fpelgrEjNPF48Uj249KNGYc'
export_file_name = 'Can_classify.pkl'

classes =  ['RecumbentBikes', 'LegAbductionMachine', 'AbBenches', 'Hyperextensionbench', 'HammerStrengthMachine', 'Handgripexerciser', 'Stationarbike', 'PecDeckMachine', 'Benchpress', 'Stretchingmachine', 'CableMachine', 'AbCoaster', 'Weights', 'CalfMachines', 'ReverseHyper', 'PowerSled', 'Preacherbench', 'LegPressMachine', 'SuspensionTrainer', 'pullupbar', 'Dumbbell', 'SpinBike', 'SmithMachine', 'PowerReack', 'dipstation', 'LatPullDownMachine', 'HackSquatMachine', 'Ankle', 'KettleBalls', 'MiniBikes', 'Handgripexerciser', 'LegCurlMachine', 'Treadmills', 'Stair']
path = Path(__file__).parent

Dic_Info = {
    
    "dipstation" : "An incredibly effective piece of equipment on which you can you can perform a variety of exercises. It has 2 arms and a large base which increases stability and prevents it from toppling over.A dip bar or station is one of the best options for increased upper body muscularity",
    "Benchpress" : "A device which can help you get that perfect rock hard chest you’ve always dreamt of. Mainly used in upper body strength training exercises along with a barbell.This is one of those gym cornerstone pieces of equipment that everyone has to go to one time or another. It is vital for building upper body strength and musculature. Find out the right way to bench from the experts.",
    "InclineBenchPress" : "The inclined bench press is a variation of the bench press, using which you can perform strength training exercises at an elevated height. The shoulders and upper chest area can be targeted.Incline bench presses develop your upper chest musculature while also being a bit more joint-friendly than the other variations.",
    "HammerStrengthMachine" : "A plate loaded device which focuses on the body’s natural path of motion. You can fully explore the advantages of converging and diverging arcs of movement.A hammer strength machine lets you lift a lot more than when you lift free weights. You need to remember not to overexert yourself.",
    "LatPullDownMachine" : "The lat pulldown machine is a strength training device with a padded seat, thigh support and a long bar hanging from an upper rod. You can work your lats using this machine.If you are not able to handle pull-ups, this could be a good alternative. You can vary your grips on the pull-down rod to work with different parts of your upper body.",
    "PecDeckMachine" : "The best machine to isolate your pectoral muscles and give them a good workout. You can perform many exercises such as chest flys, butterfly, etc. using a pec deck machine.This machines is particularly beneficial to build chest and shoulder muscles, and also enhance arms strength and stability. The upper body muscles are squeezed together, causing the pectoralis major to expand and contract. This is what builds the muscle and also toughens the tissue fibers. When the exercise on this machine is done with appropriate weights, it does prove to be quite effective.",
    "Treadmills"  : "When you think of cardio workouts, treadmills are the first to come to mind. They are used to help you achieve a walking or running motion while staying in one place. Using a treadmill regularly can help you lose weight and build strength.",
    "RecumbentBikes" : "Recumbent Bikes are designed to give you great cardiovascular workout. They differ from other stationary bikes as the rider is in a reclining position when working out. This allows the rider's weight to be distributed comfortably over a larger area.Riding a recumbent bike gives you resistance training along with cardiovascular exercise.",
    "LegAbductionMachine" : "A seated exercise machine which can provide resistance when closing and the opening the legs. You can tone your glutes with abduction and inner thigh muscles during adduction using this machine.Adduction machines are mostly used by people who have been working out for longer, continuous periods of time. Beginners should avoid using this machine to much as it could cause stiffness in the legs and always pelvic aches. Even women are usually advised to not start off with this machine as it could cause muscle pulls and minor tears to someone who isn’t well-versed with its usage.",
    "AbBenches" : "Similar to hyperextension bench, the abdominal bench targets your abs. It is mainly used in performing squats and weight training. It is highly recommended that you maintain optimal posture while working out. While it’s alright to experience some muscular fatigue or burning when using an abdominal bench, the movements should not cause intense pain. Always start slow and consult a professional before beginning a new fitness routine.",
    "Hyperextensionbench" : "",
    "Handgripexerciser" : "Mechanical handgrips are inexpensive training devices that can help you build hand strength. They are thick springs with handles on them that isolate the muscles associated with grip strength.When using mechanical handgrips, always warm up and vary your training to avoid overdoing it.",
    "Stationarbike" : "This is the granddaddy of all cardio machines, the simple stationary bike. It has been around since forever but you’ll still find them in all gyms over the country because it is that effective.Stationary bikes have had many variations over the years, some of which have been more effective, and some less so. But it has not stopped it from being a staple on every gym floor ever.",
    "Stretchingmachine" : "Stretching machines are ergonomically designed devices to make stretching movements easier and more efficient. They are ideal for athletes and also useful for preventing sport-related injuries.There are both full body stretching machines and leg stretching machines available, so it’s important to decide which one of these is better suited to your workouts.",
    "CableMachine" : "Also known as the cable crossover machine, it has a framework with grips that are attached to cables which are further connected to weights. A variety of exercises can be performed on this. This has to be one of the most versatile pieces of gym equipment apart from a dumbbell. You can find our reviews of the best cable crossover machine you can get.",
    "AbCoaster" : "The Ab coaster machine works your abs from the bottom up, by targeting those hard to reach ab muscles. It can maximize your core workout by working your abdominal muscles. Definitely enhances regular floor ab exercises and is more effective on working on the whole ab muscle, from bottom to top. It works especially on the abdominal muscles and creates core strength in weaker portions of the abs. It is also great to create core definition for aesthetic purposes and is used by athletes for during sports training.",
    "Weights" : "An essential piece of equipment in weight training with a bar and weights attached at each end. You can change the weight plates in certain dumbbells for greater versatility. The tech of dumbbells has grown and now we even have adjustable varieties that do away with having to need a rack full of dumbbells. ",
    "CalfMachines" : "The calf machine can isolate the soleus muscles of the calves and give it a good workout. Your calves get a more muscular appearance when you work the soleus muscles.This machine is basically used to do seated calf raises. Weights can be varied and the frequency depends on the person. A lot of athletes and people who require extensive legwork prefer to use calf machines to strengthen their lower legs and also increase speed and agility.",
    "ReverseHyper" : "The reverse hyperextension combines strengthening and rehab into one effective machine. The spine is gently stretched and the lumbar area is strengthened. This is definitely one of the best therapeutic machines for the back. One of the major reason for putting your back out is during heavy lifting or squatting. ",
    "PowerSled" : "",
    "Preacherbench" : "The ideal machine to build your biceps. You can increase muscle mass by lifting the barbell up and down. The machine has an elbow budding, a bar rest, and a seat. Using a preacher bench is a great way to target your biceps without putting pressure on your wrists. Another way to use this device is with a reverse grip to target the forearms.",
    "LegPressMachine" : "The ultimate machine to work your leg muscles is the leg press. You will have to push the platform of weights upwards as you lie down with your back against a support or a seat. When it comes to toning and sculpting your legs, this is a tried and tested piece of gym equipment that delivers the best results. Using a leg press is a great way to strengthen and sculpt the entire leg, and engage multiple muscle groups.",
    "SuspensionTrainer" : "It consists of several nylon bands and plastic buckles and is used to leverage your body against gravity to create resistance. Using a suspension trainer, you can perform hundreds of different exercise movements which work on various parts of the body. This was an idea by a US navy seal to come up with TRX range of suspension trainers. All you needed with this training equipment was anchor and gravity, both of which are accessible anytime.",
    "pullupbar" : "As the name suggests, this device will help you do your pull-ups better. It’s also called a chin-up bar because you pull yourself up with your chin above the bar.One of the best upper body exercises, pull-ups actually work the whole body to some extent. ", 
    "Dumbbell" : "An essential piece of equipment in weight training with a bar and weights attached at each end. You can change the weight plates in certain dumbbells for greater versatility.The tech of dumbbells has grown and now we even have adjustable varieties that do away with having to need a rack full of dumbbells. ",
    "SpinBike" : "A spin bike delivers an effective indoor workout. These bikes come with a heavyweight flywheel and cycling shoe compatibility. They also come with varying resistance levels to adjust the intensity level of the workout. Spin bikes are growing in popularity as they are an excellent way to burn calories by sweating it out. It is important not to overdo it and choose the right bike for your needs from the numerous options available.",
    "SmithMachine" : "A weight training machine which will assist you in lifting weights and in performing squats. The machine has a barbell which is fixed within steel rails allowing vertical movements. This is a great piece of equipment for intermediate and advanced lifters. But it is important to understand its limitations and use it accordingly.",
    "PowerReack" : "A piece of weight training equipment which helps you perform barbell exercises by functioning as a mechanical spotter with saddles that can rest at varied heights and bars that can go across such heights.Power racks are the cornerstone of strength building and deserve their place in every gym, commercial or indoor.", 
    "LatPullDownMachine" : "The lat pulldown machine is a strength training device with a padded seat, thigh support and a long bar hanging from an upper rod. You can work your lats using this machine. If you are not able to handle pull-ups, this could be a good alternative. You can vary your grips on the pull-down rod to work with different parts of your upper body.",
    "HackSquatMachine" : "Another fitness device that can give your legs a good workout. It is essentially a combination of leg press and squat machines. It works your quadriceps in a much more efficient way.This along with a leg press machine can help develop your lower body especially the thighs and calf sections.",
    "Ankle" : "A well-cushioned weight-bracelet for your ankle which is used to add resistance to your exercises. This is a very useful and portable method to add more to your resistance to your morning walks or jogging routine.",
    "KettleBalls" : "One of the most ancient and efficient pieces of strength training equipment is the kettlebell. Consisting of an iron ball with a handle, there are hundreds of exercises which you can do using a kettlebell. It’s important to be careful and precise about posture as these work on multiple core muscles. There are a number of exercises and movements that can be paired with the kettlebell as it is quite diverse on its own. One can easily do a full body workout with just kettlebells and different weight modulation.",
    "MiniBikes" : "A mini exercise bike is a convenient alternative for people who don’t have a lot of space. It is a portable and efficient piece of equipment that can be stashed away in a closet or even under your desk at work. It can be hard to find a mini exercise bike that is both durable and convenient to carry with you when you’re on the move.",
    "Handgripexerciser" : "Mechanical handgrips are inexpensive training devices that can help you build hand strength. They are thick springs with handles on them that isolate the muscles associated with grip strength. When using mechanical handgrips, always warm up and vary your training to avoid overdoing it.",
    "LegCurlMachine" : "The ideal machine for toning your quadriceps is the leg extension machine. You will have to sit on the machine with your legs under the pads and lift weights using your quadriceps. A lot of people have gone past leg extensions to compound exercises, but they still find them useful in certain situations.",
    "Stair" : "This is an inexpensive machine which can simulate climbing stairs which is considered a really good cardio workout. While the stair stepper may be an older machine compared to the newfangled pieces of equipment available, it can still give you that lower body burn."
}
video = {
  "dipstation" : "W8jXc1zaLuQ" ,
  "Treadmills" : "W8jXc1zaLuQ"
}
title = {
      "dipstation" : "Dip Station",
      "Benchpress" : "Bench Press",
      "InclineBenchPress" : "Incline Bench Press",
      "HammerStrengthMachine" : "Hammer Strength Machine",
      "LatPullDownMachine" : "Lat Pull-Down Machine",
      "PecDeckMachine" : "Pec Deck Machine",
      "Treadmills": "Treadmill",
      "RecumbentBikes" :"Recumbent Bikes",
      "LegAbductionMachine" : "Leg Abduction Machine",
      "AbBenches" : "Abdominal Bench",
      "Hyperextensionbench" : "",
      "Handgripexerciser" : "Hand Grip Exerciser",
      "Stationarbike" : "Stationary Bike",
      "Stretchingmachine" : "Streching Machine",
      "CableMachine" : "Cable Pulley Machine",
      "AbCoaster" : "Ab Coaster",
      "Weights" : "Dumbbells",
      "CalfMachines" : "Calf Machines",
      "ReverseHyper" : "Reverse Hyper",
      "PowerSled" : "",
      "Preacherbench" : "Preacher Bench",
      "LegPressMachine" : "Leg Press Machine",
      "SuspensionTrainer" : "Suspension Trainer",
      "pullupbar" : "Pull-up Bar",
      "Dumbbell" : "Dumb Bells",
      "SpinBike" : "Spin Bike",
      "SmithMachine" : " SMith Machine",
      "PowerReack" : "Power/Squat Racks", 
      "LatPullDownMachine" : "Lat Pull-Down Machine",
     "HackSquatMachine" : "HackSquatMachine",
     "Ankle" : "Ankle Weights",
     "KettleBalls" : "Kettlebells",
     "MiniBikes" : "Mini Exercise Bikes",
     "Handgripexerciser" : "Handgrip Exerciser",
     "LegCurlMachine" : "Leg Curl Machine",
     "Stair" : "Stair Stepper"
}

Dic_How_Often = {
        "dipstation" : "Dips work your chest and arms so you need to ideally supplement this with your arm and chest workouts. You can do this on chest and arm days but just make sure you don’t hit them too hard consecutively.",
        "Benchpress" : "If you are pushing heavy, have rest days in between so your chest can recover.",
         "InclineBenchPress" : "Inclined bench is part of the chest exercise, so the best answer would be when you have to work on your chest.",
         "HammerStrengthMachine" : "It’s a good movement to supplement your biceps/arms workout. Remember to not overdo it and give your arms about 2 days rest before you work the same muscle group.",
         "LatPullDownMachine" : "You can do lat pulldowns twice every week without fatiguing your muscles.",
         "PecDeckMachine" : "Can be used once a week, with varying weights.",
         "Treadmills" : "The frequency will depend on the level of impact of your workouts. Treadmills can be used every day, when opting for low-intensity exercise. For high-intensity runs, it is better to stick to alternate days.",
         "RecumbentBikes" : "Aim for 250 minutes of cycling per week at moderate-intensity or 100 minutes at a vigorous-intensity for weight loss.",
         "LegAbductionMachine" : "Not more than twice a week with appropriate weights.",
         "AbBenches" : "The abdominal bench can be used 2-3 times a week. It is recommended to take rest days between workouts.",
         "Hyperextensionbench" : "",
         "Handgripexerciser" : "You can do this workout 3 times a week. Start with 2 warm-up sets, followed by 3 intense sets of moderate-to-low reps.",
         "Stationarbike" : "You can safely use this every day because it’s a non-impact exercise and when used consistently over a longer period of time can burn flab and tone your body.",
         "Stretchingmachine" : "Stretching before any workout is highly recommended to avoid injuries. Spend ten minutes stretching out your muscles before beginning your exercise regime.",
         "CableMachine" : "Since there are a lot of variations you can perform on this, if you target different parts, you can use it every day.",
         "AbCoaster" : "Thrice a week",
         "Weights" : "Varies depending on the type of workout.",
         "CalfMachines" : "Can be used up to 4-6 times a week. Unless the calf muscles are sore, the machine can be used every day as well.",
         "ReverseHyper" : "If you are recovering from an injury (foot, knee, or ankle), you can regularly use this machine till you get better. For strengthening your back, this can be added to your queue during back workouts.",
         "PowerSled" : "",
         "Preacherbench" : "It is best to train your biceps using a preacher bench once a week for the best results.",
         "LegPressMachine" : "Leg press workouts should be performed 2-3 times a week. Make sure you have at least one day of rest in between.",
         "SuspensionTrainer" : "You can use this everyday if you aren’t focusing on weight training or alternate it with cardio or resistance training.",
         "pullupbar" : "Pull-ups can generally be done up to 3 times a week.",
         "Dumbbell" : "Varies depending on the type of workout.",
         "SpinBike" : "An hour of spinning 5 times a week will burn those calories and keep you fit.",
         "SmithMachine" : "It is a good idea to use a smith machine when lifting weights, as it can give you a more isolated and safer workout.",
         "PowerReack" : "You engage a lot of muscles when you use a power rack because all movements are compound exercises. Generally, if you are doing heavy movements involving a muscle group, then you rest it for at least 3 days.", 
         "LatPullDownMachine" : "You can do lat pulldowns twice every week without fatiguing your muscles.",
         "HackSquatMachine" : "You can use hack squat machines for leg days, meaning about 3 times a week, max.",
         "Ankle" : "You can use this whenever you want to add more resistance to lower body exercises or cardio sessions.",
         "KettleBalls" : "Twice a week",
         "MiniBikes" : "If weight loss is your goal then spending more time pedaling will decrease your body fat, increase your calorie burn, and help you lose more weight. Opt for at least 10 minutes of intense pedaling per day.",
         "Handgripexerciser" : "You can do this workout 3 times a week. Start with 2 warm-up sets, followed by 3 intense sets of moderate-to-low reps.", 
         "LegCurlMachine" : "Leg extensions are best done once a week because you shouldn’t tax your knees too much.",
         "Stair" : "You can use this for leg days when you are really keen to get that burn going. This machine works the thighs and calf muscles considerably."
}

Dic_Muscles_Worked = {
        "dipstation" : "Chest, triceps, front shoulders",
        "Benchpress" : "Chest, triceps, deltoids, traps and back",
        "InclineBenchPress" : "Upper chest, triceps, deltoids.",
       "HammerStrengthMachine" : "Lats, chest, middle back, shoulders and triceps",
        "LatPullDownMachine" : "Lats, deltoids, trapezius and rhomboids",
        "PecDeckMachine" : "The pectoralis major, pectoralis minor and serratus anterior.",
        "Treadmills" : "Running or walking on a treadmill primarily involves the cardiovascular system and the lower body. The other muscles involved are the hamstrings, quadriceps, calves and glutes.",
        "RecumbentBikes" :  "The quadriceps and glutes get worked when you exercise on a recumbent bike",
        "LegAbductionMachine" : "It works the adductor brevis, which runs from the upper femur down to the pelvis; the adductor longus, which stretches from the pelvis to the middle of the femur; and the adductor magnus, which also begins at the pelvis and runs down to the lower part of the femur.Also works on the pectineus and gracilis muscles.",
        "AbBenches" : "This is a very versatile piece of exercise equipment that can be used to strengthen various unique muscle groups, not just in the abdomen but throughout the body.",
        "Hyperextensionbench" : "",
        "Handgripexerciser" : "A good handgrip exerciser has the ability to produce quick gains in hand health and grip strength.",
        "Stationarbike" : "Full body cardio workout, but more towards the lower body.",
        "Stretchingmachine" : "Some machines target specific areas while others work on the entire body.",
        "CableMachine" : "Full-body workout!",
        "AbCoaster" : "Upper abs, lower abs and side obliques",
        "Weights" : "Varies depending on the type of workout.",
        "CalfMachines" : "Gastrocnemius, tibialis posterior and soleus muscles of the lower leg.",
        "ReverseHyper" : " Lower lumbar, glutes, hamstrings, and hips.",
        "PowerSled" : "",
        "Preacherbench" : "Preacher curls are used for isolating the muscles of the upper arm. They improve wrist and grip strength while targeting your biceps and forearms",
        "LegPressMachine" : "This piece of equipment primarily targets your quads, glutes, and hamstrings",
        "SuspensionTrainer" : "Full-body workout!",
        "pullupbar" : "Lats, biceps, also engages the entire body to a degree.",
        "Dumbbell" : " Varies depending on the type of workout.",
        "SpinBike" : "In addition to the leg muscles worked on a spin bike, the abdominal muscles also get a workout.",
        "SmithMachine" : "Using a smith machine bolsters the upper back, improves core strength, and stabilizes the shoulders.",
        "PowerReack" : "Varies with the type of exercise – squats, deadlifts, bench presses", 
        "LatPullDownMachine" : "Lats, deltoids, trapezius and rhomboids",
         "HackSquatMachine" : "Quadriceps, gluteus maximus",
         "Ankle" : "Lower body muscles, depending on the type of movementSometimes with really heavy ankle weights, you might experience a bit of abrasion against your skin. You can wear tall socks to circumvent this.",
         "KettleBalls" : "Works on the core, shoulders, quads, glutes, hamstrings and back muscles.",
         "MiniBikes" : "Using this device regularly can help improve your cardiovascular health and help you attain your weight loss goals. It strengthens your leg muscles and reduces total fat mass and body fat percentage.",
         "Handgripexerciser" : "A good handgrip exerciser has the ability to produce quick gains in hand health and grip strength.",
         "LegCurlMachine" : "Quadriceps, rectus femoris.",
         "Stair" : " Hips, quads, hamstrings, calves, and lower shin."
}


Dic_Tips = {
        "dipstation" : "If you want to work your chest during dips just make sure that you lean your body forwards a bit. When you want to work your triceps, you need to stay upright throughout the movement.",
        "Benchpress" : "If you want to build a good chest, remember that form is more important than poundage.",
        "InclineBenchPress" : "Most of the rules of straight bench press apply here like not hyperextending the back or not bouncing it off your chest.",
        "HammerStrengthMachine" : "The action is similar to a shoulder press but a bit more at an angle. The key is to maintain your form and lift smooth rather than jerk it.",
        "LatPullDownMachine" : "When executing this movement, make sure that you don’t hyperextend your back too much.",
        "PecDeckMachine" : "Pair up the pec deck fly along with bench press and bent over cable crossover flys for better results.",
        "Treadmills" : "Always make sure that you warm up before using the treadmill. Start with a slight incline and ensure that it is not too steep. With practice, you can work on improving your stride count.",
        "RecumbentBikes" : "Ensure that you maintain the right posture when using a recumbent bike.",
        "LegAbductionMachine" : "Always better to use this machine towards the end of a workout session. Weights should not exceed threshold of the person as it could become injurious.",
        "AbBenches" : "Whatever exercise you are doing on the abdominal bench, it is important to maintain the proper form to maximize your movements and avoid injuries.",
        "Hyperextensionbench" : "",
        "Handgripexerciser" : "A good tip to follow is to try reaching a goal of 12 consecutive reps on one gripper. Once you do this, you can start working on reaching the next level.",
        "Stationarbike" : "When starting out, find your comfort zone in the rpm range and stick with it. Find the optimal seat height so that your legs do not fully stretch when pedaling. Remember to pedal with the balls of your feet rather than the middle or the heel part.",
        "Stretchingmachine" : "When stretching, it is important to focus on using proper form and the complete range of motion",
        "CableMachine" : "From lunges to squats to rows, there are a lot of things that you can accomplish with a cable crossover machine.",
        "AbCoaster" : "Not recommended for people with weaker abdominal muscles. Definitely need to followed up by stretches and periods of rest to let the muscles relax.",
        "Weights" : "A dumbbell is one of the most effective and nifty pieces of fitness kit ever. And if you know how, you can literally exercise every part of your body just using a pair of dumbbells.",
        "CalfMachines" : "Keep increasing the weights from one alternative day to the other. This helps with pushing the core strength of the calf muscle. Avoid cardio exercises right after calf raises. Cardio before is a better option.",
        "ReverseHyper" : "Have a firm grip on the front handles and make sure that there are no wobbles when you use the machine. Make sure that you don’t swing the weighted ends too hard.",
        "PowerSled" : "",
       "Preacherbench" : "It is important not to overextend your elbows at the bottom portion of the lift. Always use a controlled motion and don’t lift the bar too fast.",
       "LegPressMachine" : "As with all other physical exercises, it is important to maintain proper form. Your knees must be at a 90-degree angle and your feet must be placed against the board. Your Legs should be in line with your knees to avoid putting strain on your joints. Also, avoid locking out your knees.",
       "SuspensionTrainer" : "One concern with suspension trainers is that you need to make sure that you are hanging the straps from stable points rather than wonky support.",
       "pullupbar" : "For increased difficulty, try pulling up from a dead hang with straight elbows. You can also add more weight through chains and barbell harnesses.",
       "Dumbbell" : "A dumbbell is one of the most effective and nifty pieces of fitness kit ever. And if you know how, you can literally exercise every part of your body just using a pair of dumbbells.",
       "SpinBike" : "Remember not to set the seat too low or position it too close to the handlebars. Maintaining the right posture is key to getting maximum benefits from your workout.",
       "SmithMachine" : "It is not advisable to use a smith machine for anything other than partial-range short duration training or calf raises.",
       "PowerReack" : "Make sure you invest in a good quality power rack which has safeties and is built to last.", 
       "LatPullDownMachine" : "When executing this movement, make sure that you don’t hyperextend your back too much.",
       "HackSquatMachine" : "Like squats, always ensure that you're pushing against the weight will all of your foot rather than just the heel or toe.",
       "Ankle" : "Sometimes with really heavy ankle weights, you might experience a bit of abrasion against your skin. You can wear tall socks to circumvent this.",
       "KettleBalls" : "Use a balanced weight and make sure your legs are wide apart while squatting to do the exercise.",
       "MiniBikes" : "Use this device when working at your desk or cycle away whilst you're watching TV, and you’ll soon start seeing positive results.",
       "Handgripexerciser" : "A good tip to follow is to try reaching a goal of 12 consecutive reps on one gripper. Once you do this, you can start working on reaching the next level.",
       "LegCurlMachine" : "You should remember never to hyper-lock your legs because that can cause stress on your knees.",
       "Stair" : "You need to remember to keep your body upright and hips centered over your legs. Keep your movements gentle and remember to push your heel into the step. Squeeze your glutes through each repetition."
}


                         
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

    

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    title_json = title[str(prediction)]
    info_json = Dic_Info[str(prediction)]
    often_json = Dic_How_Often[str(prediction)]
    muscles_json = Dic_Muscles_Worked[str(prediction)]
    tips_json = Dic_Tips[str(prediction)]
    video_json = video[str(prediction)]
 
    
    
    
    return JSONResponse({'title' : str(title_json), 'info': str(info_json), 'often' : str(often_json), 'muscles' : str(muscles_json), 'tips' : str(tips_json), 'video' : str(video_json)})
    
        

    


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
