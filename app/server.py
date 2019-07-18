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
    "BenchPress" : "A device which can help you get that perfect rock hard chest you’ve always dreamt of. Mainly used in upper body strength training exercises along with a barbell.This is one of those gym cornerstone pieces of equipment that everyone has to go to one time or another. It is vital for building upper body strength and musculature. Find out the right way to bench from the experts.",
    "InclineBenchPress" : "The inclined bench press is a variation of the bench press, using which you can perform strength training exercises at an elevated height. The shoulders and upper chest area can be targeted.Incline bench presses develop your upper chest musculature while also being a bit more joint-friendly than the other variations.",
    "HammerStrengthmachine" : "A plate loaded device which focuses on the body’s natural path of motion. You can fully explore the advantages of converging and diverging arcs of movement.A hammer strength machine lets you lift a lot more than when you lift free weights. You need to remember not to overexert yourself.",
    "LatPullDownMachine" : "The lat pulldown machine is a strength training device with a padded seat, thigh support and a long bar hanging from an upper rod. You can work your lats using this machine.If you are not able to handle pull-ups, this could be a good alternative. You can vary your grips on the pull-down rod to work with different parts of your upper body.",
    "PecDeckMachine" : "The best machine to isolate your pectoral muscles and give them a good workout. You can perform many exercises such as chest flys, butterfly, etc. using a pec deck machine.This machines is particularly beneficial to build chest and shoulder muscles, and also enhance arms strength and stability. The upper body muscles are squeezed together, causing the pectoralis major to expand and contract. This is what builds the muscle and also toughens the tissue fibers. When the exercise on this machine is done with appropriate weights, it does prove to be quite effective.",
    "Treadmill"  : "When you think of cardio workouts, treadmills are the first to come to mind. They are used to help you achieve a walking or running motion while staying in one place. Using a treadmill regularly can help you lose weight and build strength."
}

title = {
      "dipstation" : "Dip Station",
      "BenchPress" : "Bench Press",
      "InclineBenchPress" : "Incline Bench Press",
      "HammerStrengthmachine" : "Hammer Strength Machine",
      "LatPullDownMachine" : "Lat Pull-Down Machine",
      "PecDeckMachine" : "Pec Deck Machine",
      "Treadmill": "Treadmill"


}

Dic_How_Often = {
        "dipstation" : "Dips work your chest and arms so you need to ideally supplement this with your arm and chest workouts. You can do this on chest and arm days but just make sure you don’t hit them too hard consecutively.",
        "BenchPress" : "If you are pushing heavy, have rest days in between so your chest can recover.",
         "InclineBenchPress" : "Inclined bench is part of the chest exercise, so the best answer would be when you have to work on your chest.",
         "HammerStrengthmachine" : "It’s a good movement to supplement your biceps/arms workout. Remember to not overdo it and give your arms about 2 days rest before you work the same muscle group.",
         "LatPullDownMachine" : "You can do lat pulldowns twice every week without fatiguing your muscles.",
         "PecDeckMachine" : "Can be used once a week, with varying weights.",
         "Treadmill" : "The frequency will depend on the level of impact of your workouts. Treadmills can be used every day, when opting for low-intensity exercise. For high-intensity runs, it is better to stick to alternate days."

}

Dic_Muscles_Worked = {
        "dipstation" : "Chest, triceps, front shoulders",
        "BenchPress" : "Chest, triceps, deltoids, traps and back",
        "InclineBenchPress" : "Upper chest, triceps, deltoids.",
       "HammerStrengthmachine" : "Lats, chest, middle back, shoulders and triceps",
        "LatPullDownMachine" : "Lats, deltoids, trapezius and rhomboids",
        "PecDeckMachine" : "The pectoralis major, pectoralis minor and serratus anterior.",
        "Treadmill" : "Running or walking on a treadmill primarily involves the cardiovascular system and the lower body. The other muscles involved are the hamstrings, quadriceps, calves and glutes."
}

Dic_Tips = {
        "dipstation" : "If you want to work your chest during dips just make sure that you lean your body forwards a bit. When you want to work your triceps, you need to stay upright throughout the movement.",
        "BenchPress" : "If you want to build a good chest, remember that form is more important than poundage.",
        "InclineBenchPress" : "Most of the rules of straight bench press apply here like not hyperextending the back or not bouncing it off your chest.",
        "HammerStrengthmachine" : "The action is similar to a shoulder press but a bit more at an angle. The key is to maintain your form and lift smooth rather than jerk it.",
        "LatPullDownMachine" : "When executing this movement, make sure that you don’t hyperextend your back too much.",
        "PecDeckMachine" : "Pair up the pec deck fly along with bench press and bent over cable crossover flys for better results.",
        "Treadmill" : "Always make sure that you warm up before using the treadmill. Start with a slight incline and ensure that it is not too steep. With practice, you can work on improving your stride count."
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
 
    
    
    
    return JSONResponse({'title' : str(title_json), 'info': str(info_json), 'often' : str(often_json), 'muscles' : str(muscles_json), 'tips' : str(tips_json)})
    
        

    


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
