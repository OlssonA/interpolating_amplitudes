module     p0_ubaru_httbar_abbrevd43h2_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh2_qp
   implicit none
   private
   complex(ki), dimension(32), public :: abb43
   complex(ki), public :: R2d43
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
      abb43(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb43(2)=NC**(-1)
      abb43(3)=es12**(-1)
      abb43(4)=spak2l3**(-1)
      abb43(5)=spbl3k2**(-1)
      abb43(6)=spbl4k2**(-1)
      abb43(7)=spbl5k2**(-1)
      abb43(8)=sqrt(mT**2)
      abb43(9)=1.0_ki/(-mT**2+es34)
      abb43(10)=abb43(2)*c1
      abb43(10)=abb43(10)-c2
      abb43(11)=abb43(10)*abb43(1)
      abb43(12)=abb43(9)*abb43(10)
      abb43(13)=abb43(11)+abb43(12)
      abb43(14)=abb43(7)*spak2l4
      abb43(15)=abb43(14)*abb43(8)
      abb43(16)=abb43(6)*abb43(8)
      abb43(17)=abb43(16)*spak2l5
      abb43(15)=abb43(15)+abb43(17)
      abb43(15)=-spbk2k1*abb43(15)*abb43(13)
      abb43(17)=abb43(7)*spbl3k2
      abb43(18)=abb43(17)*spak2l3
      abb43(18)=abb43(18)+spak2l5
      abb43(18)=abb43(18)*abb43(6)
      abb43(18)=abb43(18)+abb43(14)
      abb43(19)=mT*spbk2k1
      abb43(20)=-abb43(19)*abb43(18)*abb43(13)
      abb43(15)=abb43(15)+abb43(20)
      abb43(15)=abb43(15)*mT
      abb43(20)=mH**2*abb43(4)*abb43(5)
      abb43(21)=abb43(20)*spak2l5
      abb43(22)=spak2l4*abb43(21)
      abb43(23)=-abb43(1)*abb43(22)*abb43(10)
      abb43(22)=-abb43(22)*abb43(12)
      abb43(22)=abb43(23)+abb43(22)
      abb43(23)=abb43(22)*spbk2k1
      abb43(15)=abb43(15)+abb43(23)
      abb43(23)=spak2l4*spal3l5
      abb43(10)=-abb43(1)*spbl3k1*abb43(10)*abb43(23)
      abb43(24)=abb43(12)*spak2l5*spal3l4
      abb43(25)=-spbl3k1*abb43(24)
      abb43(10)=abb43(10)+abb43(25)
      abb43(25)=abb43(10)+abb43(15)
      abb43(26)=2.0_ki*abb43(3)
      abb43(27)=i_*gs**4*gHT*e
      abb43(28)=abb43(27)*TR**2
      abb43(29)=abb43(26)*abb43(28)
      abb43(30)=mT**2
      abb43(26)=abb43(30)*abb43(26)
      abb43(31)=-1.0_ki/3.0_ki+abb43(26)
      abb43(31)=abb43(25)*abb43(31)*abb43(29)
      abb43(32)=abb43(8)**2
      abb43(18)=-abb43(19)*abb43(13)*abb43(32)*abb43(18)
      abb43(19)=spak2l5*abb43(6)
      abb43(14)=abb43(19)+abb43(14)
      abb43(14)=-spbk2k1*abb43(13)*abb43(14)*abb43(8)**3
      abb43(14)=abb43(14)+abb43(18)
      abb43(14)=mT*abb43(14)
      abb43(18)=spbk2k1*abb43(32)*abb43(22)
      abb43(10)=abb43(32)*abb43(10)
      abb43(10)=abb43(14)+abb43(18)+abb43(10)
      abb43(14)=abb43(3)*TR
      abb43(14)=abb43(14)**2
      abb43(18)=4.0_ki*abb43(14)
      abb43(18)=abb43(18)*abb43(27)
      abb43(10)=abb43(10)*abb43(18)
      abb43(15)=abb43(15)*abb43(18)
      abb43(19)=-abb43(25)*abb43(18)
      abb43(16)=abb43(16)*abb43(13)
      abb43(22)=mT*abb43(6)*abb43(13)
      abb43(16)=abb43(22)+abb43(16)
      abb43(16)=abb43(16)*mT
      abb43(20)=-abb43(12)*abb43(20)*spak2l4
      abb43(16)=abb43(16)-abb43(20)
      abb43(20)=abb43(3)*spbl3k1*spak1k2
      abb43(12)=abb43(12)*spal3l4
      abb43(22)=abb43(12)*abb43(20)
      abb43(22)=abb43(16)+abb43(22)
      abb43(22)=abb43(22)*abb43(29)
      abb43(25)=abb43(13)*abb43(7)*abb43(8)
      abb43(32)=mT*abb43(7)*abb43(13)
      abb43(25)=abb43(32)+abb43(25)
      abb43(25)=abb43(25)*mT
      abb43(21)=abb43(11)*abb43(21)
      abb43(21)=abb43(25)+abb43(21)
      abb43(25)=abb43(11)*spal3l5
      abb43(20)=abb43(20)*abb43(25)
      abb43(20)=abb43(21)+abb43(20)
      abb43(20)=abb43(20)*abb43(29)
      abb43(13)=abb43(13)*abb43(17)*abb43(6)
      abb43(17)=abb43(13)*abb43(28)*abb43(26)
      abb43(11)=-abb43(23)*abb43(11)
      abb43(11)=abb43(11)-abb43(24)
      abb43(11)=2.0_ki*spbk2k1*abb43(11)*abb43(14)*abb43(27)
      abb43(12)=-abb43(18)*abb43(12)
      abb43(14)=-abb43(16)*abb43(18)
      abb43(16)=-abb43(18)*abb43(25)
      abb43(21)=-abb43(21)*abb43(18)
      abb43(13)=-abb43(13)*abb43(30)*abb43(18)
      R2d43=abb43(31)
      rat2 = rat2 + R2d43
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='43' value='", &
          & R2d43, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd43h2_qp
