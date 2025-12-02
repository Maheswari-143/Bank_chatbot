async function sendMessage() {
  const input = document.getElementById('user-input');
  const message = input.value.trim();
  if (!message) return;

  const chatBox = document.getElementById('chat-box');
  chatBox.innerHTML += `<div class="message user"><div class="bubble user-bubble"><b>You:</b> ${message}</div></div>`;
  input.value = '';

  const response = await fetch('/get', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message})
  });

  const data = await response.json();
  chatBox.innerHTML += `
    <div class="message bot">
      <div class="bubble bot-bubble">
        <b>Bot:</b> ${data.response}<br>
        <small><i>Intent: ${data.intent}</i></small>
      </div>
    </div>
  `;
  chatBox.scrollTop = chatBox.scrollHeight;
}
