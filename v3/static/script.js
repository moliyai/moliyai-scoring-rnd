// document.addEventListener("DOMContentLoaded", loadJobs);
  const ageInput = document.getElementById("age");
  const ageError = document.getElementById("age-error");

  ageInput.addEventListener("input", function () {
    if (this.value < 18 || this.value > 65) {
      ageError.style.display = "block";
    } else {
      ageError.style.display = "none";
    }
  });

  const familyInput = document.getElementById("family_members");
  const familyError = document.getElementById("family-error");

  familyInput.addEventListener("input", function () {
    if (this.value < 1 || this.value > 8) {
      familyError.style.display = "block";
    } else {
      familyError.style.display = "none";
    }
  });

  const loanInput = document.getElementById("loan_amount");
  const loanError = document.getElementById("loan-error");

  loanInput.addEventListener("input", function () {
    if (this.value < 100 || this.value > 8000) {
      loanError.style.display = "block";
    } else {
      loanError.style.display = "none";
    }
  });

  const cities = {
  "наманган вилояти": [
    "Поп",
    "Норин",
    "Учкургон",
    "Янгикургон",
    "Чуст",
    "Туракургон",
    "Мингбулок",
    "Косонсой",
    "Чорток",
    "Наманган",
    "Уйчи"
  ],
  "фаргона вилояти": [
    "Фаргона",
    "Кува",
    "Олтиарик",
    "Бешарик",
    "Бувайда",
    "Дангара",
    "Богдод",
    "Тошлок",
    "Узбекистон",
    "Риштон",
    "Фуркат",
    "Ёзёвон",
    "Маргилан",
    "Кувасой",
    "Сух"
  ],
  "андижон вилояти": [
    "Асака",
    "Избоскан",
    "Хужаобод",
    "Кургонтепа",
    "Шахрихон",
    "Пахтаобод",
    "Жалолкудук",
    "Бустон",
    "Улугнор",
    "Андижон",
    "Булибоши",
    "Мархамат",
    "Олтинкул"
  ],
  "тошкент шахри": [
    "Оккургон",
    "Буқа",
    "Кибрай",
    "Бекабад",
    "Бустонлиқ",
    "Чиноз",
    "Юқоричирчиқ",
    "Қуйичирчиқ",
    "Паркент",
    "Зангиота",
    "Пискент",
    "Яшнобод",
    "Сиргали",
    "Шайхонтохур"
  ],
  "самарканд вилояти": [
    "Каттакургон",
    "Ургут"
  ],
  "сурхондарё вилояти": [
    "Кумкургон"
  ],
  "навоий вилояти": [
    "Хатирчи"
  ],
  "сирдарё вилояти": [
    "Боёвут"
  ],
  "кашкадарё вилояти": [
    "Муборак"
  ]
};


  const cityMapping = {
  // Фаргона вилояти
  "Фаргона": "фаргона шахри",
  "Кува": "кува тумани",
  "Олтиарик": "олтиарик тумани",
  "Бешарик": "бешарик тумани",
  "Дангара": "дангара тумани",
  "Богдод": "богдод тумани",
  "Тошлок": "тошлок тумани",
  "Узбекистон": "узбекистон тумани",
  "Риштон": "риштон тумани",
  "Фуркат": "фуркат тумани",
  "Ёзёвон": "ёзёвон тумани",
  "Маргилан": "маргилон шахри",
  "Кувасой": "кувасой шахри",
  "Сух": "сух тумани",

  // Андижон вилояти
  "Асака": "асака тумани",
  "Избоскан": "избоскан тумани",
  "Хужаобод": "хужаобод тумани",
  "Кургонтепа": "кургонтепа тумани",
  "Шахрихон": "шахрихон тумани",
  "Пахтаобод": "пахтаобод тумани",
  "Жалолкудук": "жалолкудук тумани",
  "Бустон": "бустон тумани",
  "Улугнор": "улугнор тумани",
  "Андижон": "андижон шахри",
  "Булибоши": "булокбоши тумани",
  "Мархамат": "мархамат тумани",
  "Олтинкул": "олтинкул тумани",

  // Наманган вилояти
  "Поп": "поп тумани",
  "Норин": "норин тумани",
  "Учкургон": "учкургон тумани",
  "Янгикургон": "янгикургон тумани",
  "Чуст": "чуст тумани",
  "Туракургон": "туракургон тумани",
  "Мингбулок": "мингбулок тумани",
  "Косонсой": "косонсой тумани",
  "Чорток": "чорток тумани",
  "Наманган": "наманган шахри",
  "Уйчи": "уйчи тумани",

  // Тошкент вилояти ва шахри
  "Оккургон": "оккургон тумани",
  "Буқа": "бука тумани",
  "Кибрай": "кибрай тумани",
  "Бекабад": "бекабад тумани",
  "Бустонлик": "бустонлик тумани",
  "Чиноз": "чиноз тумани",
  "Юкоричирчик": "юкоричирчик тумани",
  "Куйичирчик": "куйичирчик тумани",
  "Паркент": "паркент тумани",
  "Зангиота": "зангиота тумани",
  "Пискент": "пискент тумани",
  "Яшнобод": "яшнобод тумани",
  "Сиргали": "сиргали тумани",
  "Шайхонтохур": "шайхонтохур тумани",

  // Самарканд вилояти
  "Каттакургон": "каттакургон тумани",
  "Ургут": "ургут тумани",

  // Сурхондарё вилояти
  "Кумкургон": "кумкургон тумани",

  // Навоий вилояти
  "Хатирчи": "хатирчи тумани",

  // Сирдарё вилояти
  "Боёвут": "боёвут тумани",

  // Кашкадарё вилояти
  "Муборак": "муборак тумани"
};


  const regionSelect = document.getElementById("region");
  const citySelect = document.getElementById("city");

  regionSelect.addEventListener("change", function () {
    const region = this.value;
    citySelect.innerHTML = "<option value=''>Выберите город</option>";

    if (cities[region]) {
      cities[region].forEach(city => {
        const option = document.createElement("option");
        option.value = cityMapping[city] || city.toLowerCase(); 
        option.textContent = city; 
        citySelect.appendChild(option);
      });
    }
  });

  const katmInput = document.getElementById("katm");
  const katmError = document.getElementById("katm-error");

  katmInput.addEventListener("input", function () {
    if (this.value < 0 || this.value > 500) {
      katmError.style.display = "block";
    } else {
      katmError.style.display = "none";
    }
  });


  const cumulativeInput = document.getElementById("cumulative");
  const cumulativeError = document.getElementById("cumulative-error");

  cumulativeInput.addEventListener("input", function () {
    if (this.value < 0) {
      cumulativeError.style.display = "block";
    } else {
      cumulativeError.style.display = "none";
    }
  });


document.getElementById("creditForm").addEventListener("submit", function (event) {
  const katm = parseFloat(document.querySelector("input[name='katm_value']").value);
  const debtsRadio = document.querySelector("input[name='active_debts']:checked");
  const cumulative = parseFloat(document.querySelector("input[name='cumulative_days']").value);

  const debts = debtsRadio ? debtsRadio.value : null;

  // Stop factor conditions:
  // - KATM > 200
  // - Yes active debts ("yes")
  // - Cumulative days > 60

 let reasons = [];

  if (katm < 200) {
    reasons.push("Значение KATM должно быть больше 200.");
  }
  if (debts === 'yes') {
    reasons.push("У клиента не должно быть активных долгов.");
  }
  if (cumulative > 60) {
    reasons.push("Совокупное количество просроченных дней должно быть меньше 60.");
  }

  if (reasons.length > 0) {
    event.preventDefault();
    alert(`Ошибка! Форма не соответствует требованиям:\n\n${reasons.join('\n')}\n\nПожалуйста, проверьте данные и попробуйте снова.`);
  }
});

 