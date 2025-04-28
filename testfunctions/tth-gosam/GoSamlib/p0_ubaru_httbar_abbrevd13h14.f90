module     p0_ubaru_httbar_abbrevd13h14
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh14
   implicit none
   private
   complex(ki), dimension(16), public :: abb13
   complex(ki), public :: R2d13
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_color, only: TR
      use p0_ubaru_httbar_globalsl1, only: epspow
      implicit none
      abb13(1)=sqrt(mT**2)
      abb13(2)=NC**(-1)
      abb13(3)=es12**(-1)
      abb13(4)=es45**(-1)
      abb13(5)=spak2l4**(-1)
      abb13(6)=spak2l5**(-1)
      abb13(7)=c1*abb13(2)
      abb13(7)=abb13(7)-c2
      abb13(7)=abb13(7)*e*TR**2*gs**4*i_*mT*gHT*abb13(4)*abb13(3)*abb13(1)
      abb13(8)=-spak2l3*abb13(7)
      abb13(9)=abb13(8)*abb13(6)
      abb13(10)=abb13(9)*spbl4k1
      abb13(8)=abb13(8)*abb13(5)
      abb13(11)=abb13(8)*spbl5k1
      abb13(10)=abb13(10)+abb13(11)
      abb13(10)=4.0_ki*abb13(10)
      abb13(11)=-spak1k2*spbl3k1*abb13(10)
      abb13(12)=4.0_ki*spbl3k1
      abb13(13)=-abb13(8)*abb13(12)
      abb13(12)=-abb13(9)*abb13(12)
      abb13(14)=abb13(7)*abb13(5)
      abb13(15)=spbl5k1*abb13(14)
      abb13(7)=abb13(6)*abb13(7)
      abb13(16)=spbl4k1*abb13(7)
      abb13(15)=abb13(15)+abb13(16)
      abb13(15)=spak1k2*abb13(15)
      abb13(8)=spbl5l3*abb13(8)
      abb13(9)=spbl4l3*abb13(9)
      abb13(8)=abb13(9)+2.0_ki*abb13(15)+abb13(8)
      abb13(8)=4.0_ki*abb13(8)
      abb13(9)=16.0_ki*abb13(14)
      abb13(7)=16.0_ki*abb13(7)
      R2d13=0.0_ki
      rat2 = rat2 + R2d13
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='13' value='", &
          & R2d13, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd13h14
