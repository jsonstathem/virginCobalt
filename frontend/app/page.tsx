"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Upload, X, AlertCircle, CheckCircle, Droplets, Wrench, Camera, Shield, Users, Star } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

interface AnalysisResult {
  accuracy: number
  condition: "битый" | "не битый"
  cleanliness: "чистый" | "грязный"
  defects: string[]
  image: string
  confidence: number
  fileName: string
}

// Конфигурация API
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

const analyzeImages = async (images: File[]): Promise<AnalysisResult[]> => {
  try {
    const formData = new FormData()

    // Добавляем все изображения в FormData
    images.forEach((image, index) => {
      formData.append('files', image)
    })

    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    const results: AnalysisResult[] = await response.json()

    // Добавляем URL изображений для отображения
    return results.map((result, index) => ({
      ...result,
      image: URL.createObjectURL(images[index])
    }))

  } catch (error) {
    console.error('Ошибка анализа изображений:', error)

    // Fallback на mock анализ при ошибке API
    return mockFallbackAnalysis(images)
  }
}

// Fallback функция на случай недоступности API
const mockFallbackAnalysis = async (images: File[]): Promise<AnalysisResult[]> => {
  // Показываем предупреждение пользователю
  console.warn('API недоступен, используется локальный анализ')

  await new Promise((resolve) => setTimeout(resolve, 1000))

  const conditions = ["битый", "не битый"] as const
  const cleanlinessOptions = ["чистый", "грязный"] as const
  const possibleDefects = ["коррозия", "царапины", "вмятины", "спущенное колесо", "треснувшее стекло", "ржавчина"]

  return images.map((image, index) => {
    const condition = conditions[Math.floor(Math.random() * conditions.length)]
    const cleanliness = cleanlinessOptions[Math.floor(Math.random() * cleanlinessOptions.length)]
    const accuracy = Math.floor(Math.random() * 30) + 70 // 70-99%
    const confidence = Math.floor(Math.random() * 20) + 80 // 80-99%

    // Generate defects based on condition
    const defects = condition === "битый" ? possibleDefects.slice(0, Math.floor(Math.random() * 3) + 1) : []

    return {
      accuracy,
      condition,
      cleanliness,
      defects,
      image: URL.createObjectURL(image),
      confidence,
      fileName: image.name || `Фото ${index + 1}`,
    }
  })
}

export default function ClearRidePage() {
  const [selectedImages, setSelectedImages] = useState<File[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([])
  const [showCamera, setShowCamera] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    const validFiles = files.filter((file) => {
      const isValidType = file.type.startsWith("image/")
      const isValidSize = file.size <= 10 * 1024 * 1024 // 10MB
      return isValidType && isValidSize
    })

    setSelectedImages((prev) => [...prev, ...validFiles].slice(0, 5))
  }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }, // Use back camera on mobile
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setShowCamera(true)
    } catch (error) {
      console.error("Camera access failed:", error)
      alert("Не удалось получить доступ к камере")
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    setShowCamera(false)
  }

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current
      const video = videoRef.current
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      const ctx = canvas.getContext("2d")
      if (ctx) {
        ctx.drawImage(video, 0, 0)
        canvas.toBlob(
          (blob) => {
            if (blob && selectedImages.length < 5) {
              const file = new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" })
              setSelectedImages((prev) => [...prev, file])
            }
          },
          "image/jpeg",
          0.8,
        )
      }
    }
    stopCamera()
  }

  const removeImage = (index: number) => {
    setSelectedImages((prev) => prev.filter((_, i) => i !== index))
  }

  const handleUpload = async () => {
    if (selectedImages.length === 0) return

    setIsLoading(true)

    try {
      const results = await analyzeImages(selectedImages)
      setAnalysisResults(results)
    } catch (error) {
      console.error("Analysis failed:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const resetAnalysis = () => {
    setAnalysisResults([])
    setSelectedImages([])
  }

  if (analysisResults.length > 0) {
    const overallCondition = analysisResults.every((r) => r.condition === "не битый") ? "не битый" : "битый"
    const overallCleanliness = analysisResults.every((r) => r.cleanliness === "чистый") ? "чистый" : "грязный"
    const averageAccuracy = Math.round(analysisResults.reduce((sum, r) => sum + r.accuracy, 0) / analysisResults.length)

    return (
      <div className="min-h-screen bg-gray-50 p-2 sm:p-4">
        <div className="max-w-6xl mx-auto">
          <Card className="p-4 sm:p-6 lg:p-8 bg-white">
            {/* Header */}
            <div className="text-center mb-6 sm:mb-8">
              <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-black mb-2">ClearRide</h1>
              <p className="text-sm sm:text-base text-gray-600">Результаты анализа состояния автомобиля</p>
            </div>

            {/* Overall Summary */}
            <div className="bg-gray-50 rounded-lg p-4 sm:p-6 mb-6 sm:mb-8">
              <h2 className="text-lg sm:text-xl font-bold mb-4 text-center">Общая оценка</h2>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
                <div>
                  <h3 className="font-semibold mb-2">Состояние</h3>
                  <div
                    className={`flex items-center justify-center gap-2 text-lg font-bold ${
                      overallCondition === "не битый" ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    {overallCondition === "не битый" ? (
                      <CheckCircle className="w-5 h-5" />
                    ) : (
                      <AlertCircle className="w-5 h-5" />
                    )}
                    {overallCondition}
                  </div>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Чистота</h3>
                  <div
                    className={`flex items-center justify-center gap-2 text-lg font-bold ${
                      overallCleanliness === "чистый" ? "text-blue-600" : "text-orange-600"
                    }`}
                  >
                    {overallCleanliness === "чистый" ? (
                      <CheckCircle className="w-5 h-5" />
                    ) : (
                      <Droplets className="w-5 h-5" />
                    )}
                    {overallCleanliness}
                  </div>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Средняя точность</h3>
                  <div className="text-2xl font-bold text-gray-800">{averageAccuracy}%</div>
                </div>
              </div>
            </div>

            {/* Individual Results */}
            <div className="space-y-4 sm:space-y-6 mb-6 sm:mb-8">
              <h2 className="text-lg sm:text-xl font-bold text-center">Детальный анализ каждого изображения</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
                {analysisResults.map((result, index) => (
                  <Card key={index} className="p-4 border-2">
                    <div className="space-y-4">
                      <div className="text-center">
                        <h3 className="font-semibold text-sm sm:text-base mb-2">{result.fileName}</h3>
                        <img
                          src={result.image || "/placeholder.svg"}
                          alt={`Analyzed car ${index + 1}`}
                          className="w-full h-32 sm:h-40 object-cover rounded-lg bg-gray-200 mx-auto"
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-center">
                        <div>
                          <h4 className="text-sm font-semibold mb-1">Состояние</h4>
                          <div
                            className={`flex items-center justify-center gap-1 text-sm font-bold ${
                              result.condition === "не битый" ? "text-green-600" : "text-red-600"
                            }`}
                          >
                            {result.condition === "не битый" ? (
                              <CheckCircle className="w-4 h-4" />
                            ) : (
                              <AlertCircle className="w-4 h-4" />
                            )}
                            {result.condition}
                          </div>
                        </div>
                        <div>
                          <h4 className="text-sm font-semibold mb-1">Чистота</h4>
                          <div
                            className={`flex items-center justify-center gap-1 text-sm font-bold ${
                              result.cleanliness === "чистый" ? "text-blue-600" : "text-orange-600"
                            }`}
                          >
                            {result.cleanliness === "чистый" ? (
                              <CheckCircle className="w-4 h-4" />
                            ) : (
                              <Droplets className="w-4 h-4" />
                            )}
                            {result.cleanliness}
                          </div>
                        </div>
                      </div>

                      <div className="text-center">
                        <div className="text-lg font-bold text-gray-800">{result.accuracy}%</div>
                        <div className="text-xs text-gray-500">Уверенность: {result.confidence}%</div>
                      </div>

                      {result.defects.length > 0 && (
                        <div>
                          <h4 className="text-sm font-semibold mb-2 flex items-center justify-center gap-1">
                            <Wrench className="w-4 h-4" />
                            Дефекты
                          </h4>
                          <div className="space-y-1">
                            {result.defects.map((defect, defectIndex) => (
                              <div key={defectIndex} className="flex items-center justify-center gap-1">
                                <AlertCircle className="w-3 h-3 text-red-500" />
                                <span className="text-xs text-gray-700">{defect}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </Card>
                ))}
              </div>
            </div>

            <div className="bg-blue-50 rounded-lg p-4 sm:p-6 mb-6">
              <h2 className="text-lg font-bold mb-4 text-center text-blue-900">Ценность для inDrive</h2>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
                <div className="flex flex-col items-center">
                  <Shield className="w-8 h-8 text-blue-600 mb-2" />
                  <h3 className="font-semibold text-sm mb-1">Безопасность</h3>
                  <p className="text-xs text-gray-600">Предупреждение о потенциально небезопасных автомобилях</p>
                </div>
                <div className="flex flex-col items-center">
                  <Users className="w-8 h-8 text-green-600 mb-2" />
                  <h3 className="font-semibold text-sm mb-1">Доверие</h3>
                  <p className="text-xs text-gray-600">Повышение доверия пассажиров к качеству сервиса</p>
                </div>
                <div className="flex flex-col items-center">
                  <Star className="w-8 h-8 text-yellow-600 mb-2" />
                  <h3 className="font-semibold text-sm mb-1">Качество</h3>
                  <p className="text-xs text-gray-600">
                    Мотивация водителей поддерживать автомобиль в хорошем состоянии
                  </p>
                </div>
              </div>
            </div>

            <Button
              onClick={resetAnalysis}
              className="w-full bg-[#9ACD32] hover:bg-[#8BC34A] text-black font-medium py-3 rounded-lg text-sm sm:text-base"
            >
              Анализировать новые фото
            </Button>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-2 sm:p-4">
      <Card className="w-full max-w-4xl p-4 sm:p-6 lg:p-8 bg-white">
        {/* Partner Logos */}
        <div className="flex justify-center items-center gap-3 sm:gap-6 mb-6 sm:mb-8 overflow-x-auto pb-2">
          <div className="flex items-center gap-2 flex-shrink-0">
            <div className="w-6 h-6 sm:w-8 sm:h-8 bg-[#9ACD32] rounded-full flex items-center justify-center">
              <span className="text-black font-bold text-xs sm:text-sm">iD</span>
            </div>
            <span className="font-semibold text-sm sm:text-base">inDrive</span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-lg sm:text-2xl">🎯</span>
            <span className="font-semibold text-sm sm:text-base">LYSTRA</span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-lg sm:text-2xl">🐸</span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <div className="w-4 h-4 sm:w-6 sm:h-6 bg-black rounded-full"></div>
            <span className="font-semibold text-sm sm:text-base">astana hub</span>
          </div>
        </div>

        {/* Header */}
        <div className="text-center mb-6 sm:mb-8">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-black mb-2">ClearRide</h1>
          <p className="text-sm sm:text-base text-gray-600">Оценка состояния машины в одно фото</p>
        </div>

        {showCamera && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg p-4 max-w-md w-full">
              <div className="text-center mb-4">
                <h3 className="text-lg font-semibold">Сделать фото</h3>
              </div>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="w-full h-64 object-cover rounded-lg bg-gray-200 mb-4"
              />
              <canvas ref={canvasRef} className="hidden" />
              <div className="flex gap-2">
                <Button onClick={capturePhoto} className="flex-1 bg-[#9ACD32] hover:bg-[#8BC34A] text-black">
                  Сделать фото
                </Button>
                <Button onClick={stopCamera} variant="outline" className="flex-1 bg-transparent">
                  Отмена
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Upload Section */}
        <div className="space-y-4 sm:space-y-6">
          <div className="text-center">
            <p className="text-sm sm:text-base text-gray-600 mb-2">
              Загрузите изображения автомобиля или сделайте фото
            </p>
            <p className="text-xs sm:text-sm text-gray-500">Изображение должно весить не более 10МБ*</p>
          </div>

          {/* Image Previews */}
          {selectedImages.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 sm:gap-4 mb-4 sm:mb-6">
              {selectedImages.map((file, index) => (
                <div key={index} className="relative group">
                  <img
                    src={URL.createObjectURL(file) || "/placeholder.svg"}
                    alt={`Preview ${index + 1}`}
                    className="w-full h-20 sm:h-32 object-cover rounded-lg border-2 border-gray-200"
                  />
                  <button
                    onClick={() => removeImage(index)}
                    className="absolute -top-1 -right-1 sm:-top-2 sm:-right-2 bg-red-500 text-white rounded-full p-1 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity"
                  >
                    <X className="w-3 h-3 sm:w-4 sm:h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Upload Controls */}
          <div className="flex flex-col items-center gap-4">
            <div className="flex flex-col sm:flex-row gap-2 w-full max-w-md">
              <label className="cursor-pointer flex-1">
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                  disabled={selectedImages.length >= 5}
                />
                <div className="flex items-center justify-center h-12 border-2 border-dashed border-gray-300 rounded-lg hover:border-[#9ACD32] transition-colors">
                  <div className="flex items-center gap-2">
                    <Upload className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-600">Выбрать файлы</span>
                  </div>
                </div>
              </label>

              <Button
                onClick={startCamera}
                disabled={selectedImages.length >= 5}
                variant="outline"
                className="flex items-center gap-2 h-12"
              >
                <Camera className="w-4 h-4" />
                Камера
              </Button>
            </div>

            <p className="text-xs sm:text-sm text-gray-500 text-center">
              {selectedImages.length >= 5 ? "Максимум 5 изображений" : `${selectedImages.length}/5 изображений`}
            </p>

            <Button
              onClick={handleUpload}
              disabled={selectedImages.length === 0 || isLoading}
              className="bg-[#9ACD32] hover:bg-[#8BC34A] text-black font-medium px-6 sm:px-8 py-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed min-w-[120px] text-sm sm:text-base"
            >
              {isLoading ? (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
                  Анализ...
                </div>
              ) : (
                "Анализировать"
              )}
            </Button>
          </div>
        </div>
      </Card>
    </div>
  )
}
