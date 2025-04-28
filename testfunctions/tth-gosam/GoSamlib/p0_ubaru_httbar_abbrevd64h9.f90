module     p0_ubaru_httbar_abbrevd64h9
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh9
   implicit none
   private
   complex(ki), dimension(24), public :: abb64
   complex(ki), public :: R2d64
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
      abb64(1)=1.0_ki/(-mT**2+es34)
      abb64(2)=NC**(-1)
      abb64(3)=spak2l5**(-1)
      abb64(4)=spbl5k2**(-1)
      abb64(5)=sqrt(mT**2)
      abb64(6)=spbl4k2**(-1)
      abb64(7)=abb64(2)*c1
      abb64(7)=-abb64(7)+2.0_ki*c2
      abb64(8)=abb64(7)*abb64(2)
      abb64(9)=abb64(8)*spal3l4
      abb64(10)=c1*spal3l4
      abb64(9)=abb64(9)-abb64(10)
      abb64(11)=gs**4*TR**2*abb64(1)*gHT*e*i_
      abb64(12)=4.0_ki*abb64(11)
      abb64(13)=-mT**3*abb64(12)*abb64(4)*abb64(9)*spak1k2*spbl3k2*abb64(3)**2
      abb64(14)=abb64(3)*spak1k2
      abb64(15)=-abb64(2)*abb64(14)*abb64(7)*spal3l4
      abb64(16)=abb64(10)*abb64(14)
      abb64(15)=abb64(15)+abb64(16)
      abb64(12)=mT*abb64(12)*spbl3k2*abb64(15)
      abb64(11)=2.0_ki*abb64(11)
      abb64(16)=abb64(11)*mT
      abb64(17)=abb64(16)*spbl5k2*abb64(15)
      abb64(18)=-abb64(16)*spbl5l3*abb64(15)
      abb64(19)=-abb64(2)*abb64(5)*abb64(7)
      abb64(20)=c1*abb64(5)
      abb64(21)=abb64(19)+abb64(20)
      abb64(22)=spbl5k2*spak1l5
      abb64(23)=-abb64(22)*abb64(21)
      abb64(8)=abb64(8)-c1
      abb64(24)=mT*abb64(22)*abb64(8)
      abb64(23)=abb64(24)+abb64(23)
      abb64(23)=abb64(23)*abb64(11)
      abb64(7)=abb64(7)*abb64(14)
      abb64(24)=-abb64(2)*abb64(7)*abb64(5)
      abb64(20)=abb64(20)*abb64(14)
      abb64(20)=abb64(24)+abb64(20)
      abb64(24)=-mT*abb64(20)
      abb64(7)=abb64(2)*abb64(7)
      abb64(14)=-c1*abb64(14)
      abb64(7)=abb64(14)+abb64(7)
      abb64(7)=abb64(7)*abb64(5)**2
      abb64(7)=abb64(24)+abb64(7)
      abb64(7)=abb64(7)*abb64(16)
      abb64(14)=spbl3k2*abb64(6)
      abb64(22)=abb64(16)*abb64(8)*abb64(22)*abb64(14)
      abb64(20)=-mT**2*abb64(11)*abb64(14)*abb64(20)
      abb64(10)=abb64(10)*abb64(5)
      abb64(19)=spal3l4*abb64(19)
      abb64(10)=abb64(10)+abb64(19)
      abb64(10)=abb64(11)*spbl3k2*abb64(10)
      abb64(9)=-abb64(16)*abb64(3)*es12*abb64(9)
      abb64(15)=-abb64(16)*spbl3k1*abb64(15)
      abb64(19)=mT*abb64(8)
      abb64(19)=abb64(19)-abb64(21)
      abb64(11)=abb64(19)*abb64(11)
      abb64(8)=abb64(16)*abb64(14)*abb64(8)
      R2d64=0.0_ki
      rat2 = rat2 + R2d64
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='64' value='", &
          & R2d64, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd64h9
