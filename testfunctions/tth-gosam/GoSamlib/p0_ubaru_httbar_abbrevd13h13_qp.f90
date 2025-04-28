module     p0_ubaru_httbar_abbrevd13h13_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh13_qp
   implicit none
   private
   complex(ki), dimension(25), public :: abb13
   complex(ki), public :: R2d13
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_color_qp, only: TR
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      implicit none
      abb13(1)=sqrt(mT**2)
      abb13(2)=NC**(-1)
      abb13(3)=es12**(-1)
      abb13(4)=es45**(-1)
      abb13(5)=spak2l4**(-1)
      abb13(6)=spak2l5**(-1)
      abb13(7)=spak2l3**(-1)
      abb13(8)=spbl3k2**(-1)
      abb13(9)=abb13(6)*spbl4k2
      abb13(10)=abb13(5)*spbl5k2
      abb13(9)=abb13(9)+abb13(10)
      abb13(10)=abb13(2)*c1*abb13(9)
      abb13(9)=c2*abb13(9)
      abb13(9)=abb13(10)-abb13(9)
      abb13(10)=TR**2*abb13(4)*gHT*e*gs**4*mT*i_
      abb13(11)=abb13(10)*abb13(1)
      abb13(12)=4.0_ki*abb13(11)
      abb13(13)=abb13(3)*spak1k2
      abb13(14)=abb13(12)*abb13(13)
      abb13(15)=abb13(9)*abb13(14)
      abb13(16)=spak1l3*spbl5l3
      abb13(17)=abb13(5)*c1
      abb13(18)=abb13(16)*abb13(17)
      abb13(19)=spak1l3*spbl4l3
      abb13(20)=abb13(6)*c1
      abb13(21)=abb13(19)*abb13(20)
      abb13(18)=abb13(18)+abb13(21)
      abb13(18)=abb13(18)*abb13(2)
      abb13(21)=abb13(5)*c2
      abb13(16)=abb13(16)*abb13(21)
      abb13(22)=abb13(6)*c2
      abb13(19)=abb13(19)*abb13(22)
      abb13(16)=abb13(18)-abb13(16)-abb13(19)
      abb13(18)=-spak1k2*abb13(9)*abb13(8)*abb13(7)*mH**2
      abb13(19)=spak2l3*spbl3k2
      abb13(23)=-abb13(13)*abb13(19)*abb13(9)
      abb13(24)=2.0_ki*spak1k2
      abb13(24)=abb13(9)*abb13(24)
      abb13(25)=abb13(1)**2*abb13(24)*abb13(3)
      abb13(18)=abb13(25)+abb13(23)+abb13(18)-abb13(16)
      abb13(23)=4.0_ki*abb13(1)
      abb13(10)=abb13(23)*abb13(10)*abb13(18)
      abb13(16)=-abb13(24)-abb13(16)
      abb13(18)=abb13(11)*abb13(3)
      abb13(16)=8.0_ki*abb13(16)*abb13(18)
      abb13(11)=-8.0_ki*abb13(9)*abb13(13)*abb13(11)
      abb13(13)=spbl4l3*abb13(20)
      abb13(23)=spbl5l3*abb13(17)
      abb13(13)=abb13(13)+abb13(23)
      abb13(13)=abb13(2)*abb13(13)
      abb13(23)=-spbl4l3*abb13(22)
      abb13(24)=-spbl5l3*abb13(21)
      abb13(13)=abb13(13)+abb13(23)+abb13(24)
      abb13(13)=abb13(13)*abb13(14)
      abb13(14)=abb13(17)*abb13(2)
      abb13(14)=abb13(14)-abb13(21)
      abb13(17)=abb13(19)*abb13(3)
      abb13(17)=abb13(17)+2.0_ki
      abb13(19)=abb13(12)*abb13(14)*abb13(17)
      abb13(20)=abb13(20)*abb13(2)
      abb13(20)=abb13(20)-abb13(22)
      abb13(17)=abb13(12)*abb13(20)*abb13(17)
      abb13(9)=-abb13(3)*abb13(12)*spak2l3*abb13(9)
      abb13(12)=16.0_ki*abb13(18)
      abb13(14)=abb13(14)*abb13(12)
      abb13(12)=abb13(20)*abb13(12)
      R2d13=abb13(15)
      rat2 = rat2 + R2d13
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='13' value='", &
          & R2d13, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd13h13_qp
