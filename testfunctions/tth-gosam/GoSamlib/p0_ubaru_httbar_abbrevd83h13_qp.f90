module     p0_ubaru_httbar_abbrevd83h13_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh13_qp
   implicit none
   private
   complex(ki), dimension(35), public :: abb83
   complex(ki), public :: R2d83
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
      abb83(1)=NC**(-1)
      abb83(2)=spak2l4**(-1)
      abb83(3)=spbl4k2**(-1)
      abb83(4)=sqrt(mT**2)
      abb83(5)=spak2l5**(-1)
      abb83(6)=c2*abb83(5)
      abb83(7)=abb83(2)*spak1k2
      abb83(8)=abb83(7)*abb83(4)
      abb83(9)=abb83(6)*abb83(8)
      abb83(10)=abb83(3)*abb83(2)**2
      abb83(11)=abb83(10)*spak1k2
      abb83(12)=spbl5k2*c2
      abb83(13)=-abb83(11)*abb83(12)*abb83(4)
      abb83(9)=abb83(9)+abb83(13)
      abb83(13)=spbl5k2*spak1k2
      abb83(14)=abb83(10)*abb83(13)
      abb83(15)=abb83(7)*abb83(5)
      abb83(14)=-abb83(15)+abb83(14)
      abb83(16)=abb83(1)*c1
      abb83(14)=abb83(14)*abb83(16)*abb83(4)
      abb83(9)=2.0_ki*abb83(9)+abb83(14)
      abb83(9)=abb83(1)*abb83(9)
      abb83(14)=2.0_ki*abb83(12)
      abb83(17)=abb83(14)*abb83(11)
      abb83(10)=abb83(10)*abb83(16)
      abb83(13)=-abb83(13)*abb83(10)
      abb83(13)=abb83(17)+abb83(13)
      abb83(13)=abb83(1)*abb83(13)
      abb83(17)=abb83(12)*NC
      abb83(18)=-abb83(17)*abb83(11)
      abb83(13)=abb83(18)+abb83(13)
      abb83(13)=mT*abb83(13)
      abb83(18)=c2*NC
      abb83(19)=abb83(15)*abb83(18)
      abb83(20)=-abb83(4)*abb83(19)
      abb83(21)=abb83(17)*abb83(4)
      abb83(22)=abb83(21)*abb83(11)
      abb83(9)=abb83(13)+abb83(9)+abb83(20)+abb83(22)
      abb83(9)=mT*abb83(9)
      abb83(13)=2.0_ki*abb83(6)
      abb83(20)=abb83(13)*abb83(7)
      abb83(22)=abb83(15)*abb83(16)
      abb83(20)=abb83(20)-abb83(22)
      abb83(20)=abb83(20)*abb83(1)
      abb83(19)=abb83(20)-abb83(19)
      abb83(20)=abb83(4)**2
      abb83(22)=-abb83(20)*abb83(19)
      abb83(9)=abb83(9)+abb83(22)
      abb83(9)=mT*abb83(9)
      abb83(22)=abb83(16)*spbl5k2
      abb83(14)=abb83(22)-abb83(14)
      abb83(22)=abb83(14)*abb83(1)
      abb83(23)=abb83(17)+abb83(22)
      abb83(8)=-abb83(8)*abb83(23)
      abb83(9)=abb83(9)-abb83(8)
      abb83(24)=TR**2*gHT*e*i_*gs**4
      abb83(25)=abb83(24)*mT
      abb83(25)=4.0_ki*abb83(25)
      abb83(9)=abb83(9)*abb83(25)
      abb83(11)=abb83(11)*abb83(6)
      abb83(10)=abb83(10)*spak1k2*abb83(5)
      abb83(10)=-2.0_ki*abb83(11)+abb83(10)
      abb83(10)=abb83(1)*abb83(10)
      abb83(11)=NC*abb83(11)
      abb83(10)=abb83(11)+abb83(10)
      abb83(10)=8.0_ki*abb83(10)*mT**4*abb83(24)
      abb83(11)=abb83(8)*abb83(25)
      abb83(26)=-mT*abb83(7)*abb83(23)
      abb83(8)=-2.0_ki*abb83(8)+abb83(26)
      abb83(8)=abb83(8)*abb83(25)
      abb83(25)=mT**2
      abb83(26)=-4.0_ki*abb83(19)*abb83(25)*abb83(24)
      abb83(27)=abb83(7)*abb83(18)
      abb83(28)=abb83(7)*abb83(16)
      abb83(7)=2.0_ki*abb83(7)
      abb83(7)=-c2*abb83(7)
      abb83(7)=abb83(7)+abb83(28)
      abb83(7)=abb83(1)*abb83(7)
      abb83(7)=abb83(27)+abb83(7)
      abb83(7)=abb83(25)*abb83(7)*abb83(3)*spbl5k2**2
      abb83(27)=abb83(23)*spak1l4*spbl5l4
      abb83(7)=abb83(7)+abb83(27)
      abb83(24)=2.0_ki*abb83(24)
      abb83(7)=abb83(7)*abb83(24)
      abb83(27)=abb83(2)*abb83(4)
      abb83(14)=abb83(1)*abb83(27)*abb83(14)
      abb83(28)=abb83(21)*abb83(2)
      abb83(14)=abb83(14)+abb83(28)
      abb83(28)=abb83(24)*mT
      abb83(29)=-abb83(28)*spak1l5*abb83(14)
      abb83(27)=abb83(27)*spak1l4
      abb83(30)=abb83(4)*abb83(5)
      abb83(31)=abb83(30)*spak1l5
      abb83(27)=abb83(27)-abb83(31)
      abb83(31)=abb83(27)*abb83(12)
      abb83(32)=spak1l4*spbl4k2
      abb83(6)=abb83(32)*abb83(6)
      abb83(33)=-abb83(4)*abb83(6)
      abb83(33)=abb83(33)-abb83(31)
      abb83(34)=abb83(30)*abb83(32)
      abb83(27)=spbl5k2*abb83(27)
      abb83(27)=abb83(34)+abb83(27)
      abb83(27)=abb83(1)*c1*abb83(27)
      abb83(27)=2.0_ki*abb83(33)+abb83(27)
      abb83(27)=abb83(1)*abb83(27)
      abb83(33)=abb83(15)*spal4l5
      abb83(12)=abb83(12)*abb83(33)
      abb83(6)=abb83(6)+abb83(12)
      abb83(12)=abb83(32)*abb83(5)
      abb83(35)=-spbl5k2*abb83(33)
      abb83(35)=-abb83(12)+abb83(35)
      abb83(35)=abb83(35)*abb83(16)
      abb83(6)=2.0_ki*abb83(6)+abb83(35)
      abb83(6)=abb83(1)*abb83(6)
      abb83(12)=-abb83(18)*abb83(12)
      abb83(17)=-abb83(17)*abb83(33)
      abb83(6)=abb83(6)+abb83(12)+abb83(17)
      abb83(6)=mT*abb83(6)
      abb83(12)=abb83(18)*abb83(34)
      abb83(17)=NC*abb83(31)
      abb83(6)=abb83(6)+abb83(27)+abb83(12)+abb83(17)
      abb83(6)=mT*abb83(6)
      abb83(12)=abb83(32)*abb83(23)
      abb83(6)=abb83(6)+abb83(12)
      abb83(6)=abb83(6)*abb83(24)
      abb83(12)=abb83(25)*abb83(24)
      abb83(17)=abb83(19)*abb83(12)
      abb83(12)=abb83(12)*abb83(23)*abb83(15)*spak1l5
      abb83(14)=abb83(28)*spak2l5*abb83(14)
      abb83(15)=abb83(23)*abb83(24)
      abb83(19)=abb83(20)*abb83(23)
      abb83(20)=-abb83(4)*abb83(22)
      abb83(20)=-abb83(21)+abb83(20)
      abb83(20)=mT*abb83(20)
      abb83(19)=abb83(20)+abb83(19)
      abb83(19)=abb83(19)*abb83(24)
      abb83(20)=abb83(30)*abb83(18)
      abb83(21)=abb83(16)*abb83(30)
      abb83(22)=-abb83(4)*abb83(13)
      abb83(21)=abb83(22)+abb83(21)
      abb83(21)=abb83(1)*abb83(21)
      abb83(20)=abb83(20)+abb83(21)
      abb83(16)=-abb83(5)*abb83(16)
      abb83(13)=abb83(13)+abb83(16)
      abb83(13)=abb83(1)*abb83(13)
      abb83(16)=-abb83(5)*abb83(18)
      abb83(13)=abb83(16)+abb83(13)
      abb83(13)=mT*abb83(13)
      abb83(13)=2.0_ki*abb83(20)+abb83(13)
      abb83(13)=mT*abb83(13)
      abb83(13)=abb83(13)+abb83(23)
      abb83(13)=abb83(13)*abb83(24)
      R2d83=0.0_ki
      rat2 = rat2 + R2d83
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='83' value='", &
          & R2d83, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd83h13_qp
