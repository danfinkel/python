module Rouge
  module Formatters
    class HTMLWrapper
      def initialize(open, formatter, close)
        @open = open
        @formatter = formatter
        @close = close
      end
    end
  end
end
