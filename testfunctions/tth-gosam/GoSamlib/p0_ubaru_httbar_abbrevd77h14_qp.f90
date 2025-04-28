module     p0_ubaru_httbar_abbrevd77h14_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh14_qp
   implicit none
   private
   complex(ki), dimension(38), public :: abb77
   complex(ki), public :: R2d77
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
      abb77(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb77(2)=NC**(-1)
      abb77(3)=es12**(-1)
      abb77(4)=1.0_ki/(-mT**2+es34)
      abb77(5)=spak2l4**(-1)
      abb77(6)=spak2l3**(-1)
      abb77(7)=spbl3k2**(-1)
      abb77(8)=spak2l5**(-1)
      abb77(9)=sqrt(mT**2)
      abb77(10)=abb77(2)**2
      abb77(11)=abb77(10)-1.0_ki
      abb77(11)=abb77(11)*c1
      abb77(12)=abb77(2)-NC
      abb77(12)=abb77(12)*c2
      abb77(11)=abb77(11)-abb77(12)
      abb77(12)=spak2l3*spbl4l3
      abb77(13)=abb77(12)*spbl5k1
      abb77(14)=-abb77(4)*abb77(13)*abb77(11)
      abb77(15)=spak2l3*spbl5l3
      abb77(16)=abb77(15)*spbl4k1
      abb77(17)=-abb77(1)*abb77(16)*abb77(11)
      abb77(14)=abb77(14)+abb77(17)
      abb77(17)=TR**2*gHT*e*i_*gs**4
      abb77(18)=abb77(17)*abb77(3)
      abb77(14)=abb77(14)*abb77(18)
      abb77(13)=abb77(13)*abb77(4)
      abb77(16)=abb77(16)*abb77(1)
      abb77(13)=abb77(13)+abb77(16)
      abb77(16)=c2*abb77(2)
      abb77(19)=abb77(10)*c1
      abb77(16)=abb77(16)-abb77(19)
      abb77(19)=2.0_ki*abb77(17)*abb77(16)*abb77(13)
      abb77(20)=2.0_ki*abb77(3)
      abb77(17)=abb77(20)*abb77(17)
      abb77(20)=c2*NC
      abb77(20)=abb77(20)-c1
      abb77(13)=abb77(17)*abb77(20)*abb77(13)
      abb77(21)=spbl5k1*spbl4l3
      abb77(22)=abb77(4)*abb77(11)*abb77(21)*spak1k2
      abb77(23)=spbl4k1*spbl5l3
      abb77(24)=abb77(1)*abb77(11)*abb77(23)*spak1k2
      abb77(22)=abb77(22)+abb77(24)
      abb77(22)=abb77(22)*abb77(17)
      abb77(24)=abb77(17)*abb77(4)
      abb77(25)=abb77(24)*abb77(11)*abb77(12)*spbk2k1
      abb77(24)=abb77(24)*spbl4l3
      abb77(26)=abb77(16)*abb77(24)
      abb77(27)=abb77(17)*abb77(1)
      abb77(11)=abb77(27)*abb77(11)*abb77(15)*spbk2k1
      abb77(27)=abb77(27)*spbl5l3
      abb77(16)=abb77(16)*abb77(27)
      abb77(28)=spak1k2*NC
      abb77(29)=2.0_ki*spak1k2
      abb77(30)=abb77(29)*abb77(2)
      abb77(28)=abb77(28)-abb77(30)
      abb77(31)=abb77(5)*abb77(9)
      abb77(32)=abb77(31)*spbl5k1
      abb77(33)=abb77(8)*spbl4k1
      abb77(34)=abb77(33)*abb77(9)
      abb77(32)=abb77(32)+abb77(34)
      abb77(28)=c2*abb77(32)*abb77(28)
      abb77(29)=abb77(29)*abb77(10)
      abb77(34)=abb77(29)-spak1k2
      abb77(34)=abb77(34)*c1
      abb77(32)=-abb77(32)*abb77(34)
      abb77(28)=abb77(28)-abb77(32)
      abb77(32)=abb77(1)+abb77(4)
      abb77(28)=abb77(28)*abb77(32)
      abb77(35)=abb77(8)*spak2l3
      abb77(36)=abb77(35)*spbl3k1
      abb77(36)=abb77(36)+spbl5k1
      abb77(36)=abb77(36)*abb77(5)
      abb77(33)=abb77(36)+abb77(33)
      abb77(36)=spak1k2*NC*abb77(33)
      abb77(37)=abb77(33)*abb77(30)
      abb77(36)=abb77(36)-abb77(37)
      abb77(36)=abb77(36)*c2
      abb77(33)=abb77(33)*abb77(34)
      abb77(33)=abb77(36)+abb77(33)
      abb77(34)=abb77(32)*mT
      abb77(33)=abb77(33)*abb77(34)
      abb77(28)=abb77(33)+abb77(28)
      abb77(28)=mT*abb77(28)
      abb77(12)=abb77(12)*spbl5k2
      abb77(21)=abb77(21)*spak1l3
      abb77(33)=mH**2*abb77(6)*abb77(7)
      abb77(36)=abb77(33)*spak1k2
      abb77(37)=spbl4k2*abb77(36)*spbl5k1
      abb77(12)=abb77(37)+abb77(12)-abb77(21)
      abb77(21)=abb77(33)*spbl4k2
      abb77(37)=abb77(21)*spbl5k1
      abb77(38)=abb77(29)*abb77(37)
      abb77(38)=abb77(38)-abb77(12)
      abb77(38)=c1*abb77(38)
      abb77(12)=NC*abb77(12)
      abb77(37)=-abb77(30)*abb77(37)
      abb77(12)=abb77(37)+abb77(12)
      abb77(12)=c2*abb77(12)
      abb77(12)=abb77(38)+abb77(12)
      abb77(12)=abb77(4)*abb77(12)
      abb77(15)=abb77(15)*spbl4k2
      abb77(36)=spbl5k2*abb77(36)*spbl4k1
      abb77(23)=abb77(23)*spak1l3
      abb77(15)=-abb77(23)+abb77(15)+abb77(36)
      abb77(23)=abb77(33)*spbl5k2
      abb77(33)=abb77(23)*spbl4k1
      abb77(29)=abb77(29)*abb77(33)
      abb77(29)=abb77(29)-abb77(15)
      abb77(29)=c1*abb77(29)
      abb77(15)=NC*abb77(15)
      abb77(30)=-abb77(30)*abb77(33)
      abb77(15)=abb77(30)+abb77(15)
      abb77(15)=c2*abb77(15)
      abb77(15)=abb77(29)+abb77(15)
      abb77(15)=abb77(1)*abb77(15)
      abb77(12)=abb77(28)+abb77(12)+abb77(15)
      abb77(12)=abb77(12)*abb77(18)
      abb77(15)=abb77(20)*abb77(24)
      abb77(18)=abb77(20)*abb77(27)
      abb77(10)=abb77(10)+1.0_ki
      abb77(10)=abb77(10)*c1
      abb77(20)=abb77(2)+NC
      abb77(20)=abb77(20)*c2
      abb77(10)=abb77(10)-abb77(20)
      abb77(20)=-abb77(32)*abb77(31)*abb77(10)
      abb77(24)=-abb77(34)*abb77(5)*abb77(10)
      abb77(20)=abb77(24)+abb77(20)
      abb77(20)=mT*abb77(20)
      abb77(21)=-abb77(4)*abb77(21)*abb77(10)
      abb77(20)=abb77(21)+abb77(20)
      abb77(20)=abb77(20)*abb77(17)
      abb77(21)=-abb77(32)*abb77(10)*abb77(8)*abb77(9)
      abb77(24)=-abb77(34)*abb77(8)*abb77(10)
      abb77(21)=abb77(24)+abb77(21)
      abb77(21)=mT*abb77(21)
      abb77(23)=-abb77(1)*abb77(23)*abb77(10)
      abb77(21)=abb77(23)+abb77(21)
      abb77(21)=abb77(21)*abb77(17)
      abb77(10)=-mT**2*abb77(17)*abb77(32)*abb77(10)*abb77(35)*abb77(5)
      R2d77=abb77(14)
      rat2 = rat2 + R2d77
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='77' value='", &
          & R2d77, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd77h14_qp
