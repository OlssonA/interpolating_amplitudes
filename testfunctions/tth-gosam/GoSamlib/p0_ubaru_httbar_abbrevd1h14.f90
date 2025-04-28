module     p0_ubaru_httbar_abbrevd1h14
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh14
   implicit none
   private
   complex(ki), dimension(24), public :: abb1
   complex(ki), public :: R2d1
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
      abb1(1)=1.0_ki/(-mT**2+es34)
      abb1(2)=NC**(-1)
      abb1(3)=es12**(-1)
      abb1(4)=sqrt(mT**2)
      abb1(5)=spak2l4**(-1)
      abb1(6)=spak2l3**(-1)
      abb1(7)=spbl3k2**(-1)
      abb1(8)=spak2l5**(-1)
      abb1(9)=gs**4*i_*e*gHT*TR**2*abb1(3)*abb1(1)
      abb1(10)=c1*abb1(9)*abb1(2)**2
      abb1(9)=c2*abb1(9)*abb1(2)
      abb1(9)=abb1(10)-abb1(9)
      abb1(10)=abb1(9)*spak2l3*spbl5k1
      abb1(11)=-spbl4l3*abb1(10)
      abb1(12)=2.0_ki*spbl4l3
      abb1(10)=-abb1(12)*abb1(4)**2*abb1(10)
      abb1(13)=abb1(9)*abb1(12)
      abb1(14)=spak1k2*spbl5k1
      abb1(15)=-abb1(14)*abb1(13)
      abb1(16)=abb1(4)+mT
      abb1(16)=-abb1(4)*abb1(16)*abb1(9)
      abb1(17)=-spbl4k1*abb1(16)
      abb1(18)=abb1(4)*mT
      abb1(19)=-abb1(18)*abb1(9)
      abb1(20)=spak2l3*abb1(5)
      abb1(21)=abb1(19)*abb1(20)
      abb1(22)=-spbl3k1*abb1(21)
      abb1(17)=abb1(17)+abb1(22)
      abb1(17)=4.0_ki*abb1(17)
      abb1(22)=4.0_ki*spbl5k1
      abb1(16)=abb1(16)*abb1(22)
      abb1(21)=abb1(21)*abb1(22)
      abb1(22)=mT**2
      abb1(18)=abb1(22)+abb1(18)
      abb1(18)=abb1(18)*abb1(9)
      abb1(23)=abb1(18)*abb1(5)
      abb1(24)=abb1(6)*abb1(7)*abb1(9)*spbl4k2*mH**2
      abb1(23)=abb1(24)+abb1(23)
      abb1(14)=-abb1(14)*abb1(23)
      abb1(12)=abb1(8)*abb1(19)*abb1(12)*spak2l3
      abb1(12)=abb1(12)+abb1(14)
      abb1(12)=2.0_ki*abb1(12)
      abb1(14)=-2.0_ki*abb1(23)
      abb1(19)=2.0_ki*abb1(8)
      abb1(18)=-abb1(18)*abb1(19)
      abb1(9)=-abb1(20)*abb1(19)*abb1(22)*abb1(9)
      R2d1=abb1(11)
      rat2 = rat2 + R2d1
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='1' value='", &
          & R2d1, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd1h14
