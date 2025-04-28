module     p0_ubaru_httbar_abbrevd58h14_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh14_qp
   implicit none
   private
   complex(ki), dimension(29), public :: abb58
   complex(ki), public :: R2d58
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
      abb58(1)=sqrt(mT**2)
      abb58(2)=NC**(-1)
      abb58(3)=es12**(-1)
      abb58(4)=spak2l4**(-1)
      abb58(5)=spak2l5**(-1)
      abb58(6)=spbl5k2**(-1)
      abb58(7)=spbl4k2**(-1)
      abb58(8)=i_*e*gHT*abb58(3)*TR**2*gs**4
      abb58(9)=abb58(8)*spak2l3
      abb58(10)=abb58(9)*abb58(2)
      abb58(11)=abb58(1)**2
      abb58(12)=abb58(11)*abb58(10)
      abb58(13)=mT*abb58(1)
      abb58(14)=abb58(13)*abb58(9)
      abb58(15)=abb58(2)*abb58(14)
      abb58(12)=abb58(12)+abb58(15)
      abb58(12)=c2*abb58(12)
      abb58(15)=abb58(2)**2
      abb58(16)=abb58(9)*abb58(15)
      abb58(17)=-abb58(11)*abb58(16)
      abb58(14)=-abb58(15)*abb58(14)
      abb58(14)=abb58(17)+abb58(14)
      abb58(14)=c1*abb58(14)
      abb58(12)=abb58(12)+abb58(14)
      abb58(14)=spbl5l3*spbl4k1
      abb58(12)=abb58(12)*abb58(14)
      abb58(17)=abb58(15)*abb58(8)
      abb58(18)=abb58(17)*c1
      abb58(8)=abb58(8)*abb58(2)
      abb58(19)=abb58(8)*c2
      abb58(18)=abb58(18)-abb58(19)
      abb58(19)=-abb58(11)*abb58(18)
      abb58(20)=spbl5l4*spbl5k1
      abb58(21)=-abb58(19)*abb58(20)
      abb58(22)=c2*abb58(2)
      abb58(15)=abb58(15)*c1
      abb58(15)=abb58(15)-abb58(22)
      abb58(9)=abb58(9)*abb58(15)
      abb58(15)=-abb58(13)*abb58(9)
      abb58(22)=abb58(15)*spbl5l3
      abb58(23)=abb58(4)*spbl5k1
      abb58(24)=abb58(23)*abb58(22)
      abb58(21)=abb58(21)+abb58(24)
      abb58(21)=spak2l5*abb58(21)
      abb58(24)=spbl4l3*spbl5k1
      abb58(25)=abb58(24)*abb58(11)*abb58(9)
      abb58(26)=spbl5l4*spbl4k1
      abb58(27)=abb58(26)*spak2l4
      abb58(28)=-abb58(19)*abb58(27)
      abb58(15)=abb58(15)*spbl3k1
      abb58(29)=-spbl5l4*abb58(15)
      abb58(12)=abb58(29)+abb58(28)+abb58(25)+abb58(12)+abb58(21)
      abb58(12)=2.0_ki*abb58(12)
      abb58(21)=spak2l5*abb58(20)
      abb58(21)=abb58(27)+abb58(21)
      abb58(21)=abb58(18)*abb58(21)
      abb58(14)=abb58(24)-abb58(14)
      abb58(14)=abb58(9)*abb58(14)
      abb58(14)=abb58(14)+abb58(21)
      abb58(14)=2.0_ki*abb58(14)
      abb58(21)=2.0_ki*abb58(9)
      abb58(20)=-abb58(20)*abb58(21)
      abb58(21)=-abb58(26)*abb58(21)
      abb58(11)=3.0_ki*abb58(11)+2.0_ki*abb58(13)
      abb58(17)=-c1*abb58(17)*abb58(11)
      abb58(8)=c2*abb58(8)*abb58(11)
      abb58(8)=abb58(8)+abb58(17)
      abb58(8)=spbl4k1*abb58(8)
      abb58(11)=-abb58(13)*abb58(18)
      abb58(13)=spak2l5*abb58(11)*abb58(23)
      abb58(8)=abb58(8)+2.0_ki*abb58(13)
      abb58(8)=2.0_ki*abb58(8)
      abb58(13)=-2.0_ki*spbl4k1*abb58(18)
      abb58(17)=spbl5k1*abb58(19)
      abb58(15)=-abb58(5)*abb58(15)
      abb58(15)=-3.0_ki*abb58(17)+abb58(15)
      abb58(15)=2.0_ki*abb58(15)
      abb58(17)=2.0_ki*spbl5k1
      abb58(17)=abb58(18)*abb58(17)
      abb58(18)=spbl5k2*abb58(7)*abb58(4)*spbl4k1
      abb58(19)=-spbl4k2*abb58(6)*abb58(5)*spbl5k1
      abb58(18)=abb58(19)+abb58(18)-abb58(23)
      abb58(9)=-abb58(18)*abb58(9)*mT**2
      abb58(16)=-c1*abb58(16)
      abb58(10)=c2*abb58(10)
      abb58(10)=abb58(10)+abb58(16)
      abb58(16)=mT-abb58(1)
      abb58(10)=abb58(5)*spbl4k1*mT*abb58(16)*abb58(10)
      abb58(9)=abb58(10)+abb58(9)
      abb58(9)=2.0_ki*abb58(9)
      abb58(10)=2.0_ki*spbl5l4
      abb58(10)=abb58(11)*abb58(10)
      abb58(16)=abb58(4)*abb58(22)
      abb58(10)=abb58(10)+abb58(16)
      abb58(10)=2.0_ki*abb58(10)
      abb58(11)=4.0_ki*abb58(11)
      abb58(16)=abb58(4)*abb58(11)
      abb58(11)=abb58(5)*abb58(11)
      R2d58=0.0_ki
      rat2 = rat2 + R2d58
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='58' value='", &
          & R2d58, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd58h14_qp
