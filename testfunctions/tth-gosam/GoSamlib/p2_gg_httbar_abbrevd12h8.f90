module     p2_gg_httbar_abbrevd12h8
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh8
   implicit none
   private
   complex(ki), dimension(33), public :: abb12
   complex(ki), public :: R2d12
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb12(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb12(2)=NC**(-1)
      abb12(3)=es12**(-1)
      abb12(4)=spak2l5**(-1)
      abb12(5)=spbl4k2**(-1)
      abb12(6)=spak2l3**(-1)
      abb12(7)=spbl3k2**(-1)
      abb12(8)=sqrt(mT**2)
      abb12(9)=mT**3
      abb12(10)=spbe2e1*spae1e2*gs**4*i_*TR*e*gHT*abb12(2)*abb12(1)
      abb12(11)=abb12(9)*abb12(10)
      abb12(12)=mT**2
      abb12(13)=abb12(12)*abb12(10)
      abb12(14)=abb12(13)*abb12(8)
      abb12(11)=abb12(14)+abb12(11)
      abb12(14)=c1-c2
      abb12(15)=abb12(4)*abb12(5)
      abb12(11)=-abb12(15)*abb12(11)*abb12(14)
      abb12(16)=abb12(14)*abb12(10)*mT
      abb12(17)=spbl5k2*abb12(5)
      abb12(18)=mH**2*abb12(7)*abb12(6)
      abb12(19)=abb12(17)*abb12(18)
      abb12(20)=abb12(16)*abb12(19)
      abb12(21)=spak2l4*spbl5k2
      abb12(22)=spbl5k1*spak1l4
      abb12(21)=abb12(21)-abb12(22)
      abb12(10)=abb12(10)*abb12(3)
      abb12(22)=abb12(8)*abb12(10)
      abb12(23)=mT*abb12(10)
      abb12(24)=abb12(22)+abb12(23)
      abb12(24)=-abb12(24)*abb12(14)
      abb12(21)=abb12(21)*abb12(24)
      abb12(20)=abb12(21)+abb12(11)-abb12(20)
      abb12(21)=spbl3k2*spak2l4
      abb12(25)=spbl3k1*spak1l4
      abb12(21)=abb12(21)-abb12(25)
      abb12(25)=-abb12(23)*abb12(14)
      abb12(26)=spak2l3*abb12(4)
      abb12(27)=abb12(25)*abb12(26)
      abb12(28)=abb12(27)*abb12(21)
      abb12(29)=spbl5l3*abb12(5)
      abb12(30)=abb12(29)*abb12(25)
      abb12(31)=spak1l3*spbk2k1
      abb12(32)=abb12(30)*abb12(31)
      abb12(28)=abb12(32)+abb12(28)+abb12(20)
      abb12(28)=1.0_ki/4.0_ki*abb12(28)
      abb12(32)=abb12(8)**2
      abb12(20)=abb12(32)*abb12(20)
      abb12(21)=-abb12(26)*abb12(21)
      abb12(31)=-abb12(29)*abb12(31)
      abb12(21)=abb12(31)+abb12(21)
      abb12(21)=-abb12(21)*abb12(32)*abb12(25)
      abb12(20)=abb12(20)+abb12(21)
      abb12(20)=1.0_ki/2.0_ki*abb12(20)
      abb12(13)=abb12(13)*abb12(3)
      abb12(21)=abb12(23)*abb12(8)
      abb12(13)=abb12(21)+abb12(13)
      abb12(13)=-abb12(14)*abb12(13)*abb12(8)
      abb12(21)=abb12(13)*abb12(17)
      abb12(12)=abb12(22)*abb12(12)
      abb12(23)=-abb12(12)*abb12(14)
      abb12(26)=abb12(23)*abb12(26)
      abb12(31)=spbl3k2*abb12(5)*abb12(26)
      abb12(21)=abb12(21)+abb12(31)
      abb12(17)=abb12(16)*abb12(17)
      abb12(22)=-abb12(22)*abb12(14)
      abb12(31)=abb12(22)*spbl5k2
      abb12(32)=2.0_ki*spak2l4
      abb12(33)=-abb12(31)*abb12(32)
      abb12(17)=abb12(17)+abb12(33)
      abb12(17)=abb12(17)*abb12(18)
      abb12(33)=abb12(13)*abb12(4)
      abb12(32)=-abb12(33)*abb12(32)
      abb12(11)=abb12(17)+abb12(32)-abb12(11)+2.0_ki*abb12(21)
      abb12(9)=abb12(9)*abb12(10)
      abb12(9)=abb12(9)+abb12(12)
      abb12(9)=-abb12(15)*abb12(9)*abb12(14)
      abb12(10)=abb12(19)*abb12(25)
      abb12(9)=abb12(9)+abb12(10)
      abb12(10)=-2.0_ki*abb12(9)
      abb12(12)=abb12(22)*spbl5l3
      abb12(14)=spak2l3*abb12(12)
      abb12(17)=abb12(31)*abb12(18)
      abb12(17)=abb12(17)+abb12(33)
      abb12(18)=-spak1k2*abb12(17)
      abb12(19)=-spak1l3*abb12(12)
      abb12(18)=abb12(19)+abb12(18)
      abb12(16)=abb12(5)*abb12(16)
      abb12(19)=-spak2l4*abb12(22)
      abb12(16)=1.0_ki/2.0_ki*abb12(16)+abb12(19)
      abb12(16)=spbl5l3*abb12(16)
      abb12(12)=spak1l4*abb12(12)
      abb12(19)=1.0_ki/2.0_ki*abb12(24)
      abb12(21)=spbk2k1*spak1l4
      abb12(22)=abb12(19)*abb12(21)
      abb12(24)=1.0_ki/2.0_ki*abb12(27)
      abb12(21)=abb12(24)*abb12(21)
      abb12(17)=spak1l4*abb12(17)
      abb12(27)=abb12(5)*abb12(13)
      abb12(31)=spak2l4*abb12(19)
      abb12(27)=abb12(27)+abb12(31)
      abb12(27)=spbk2k1*abb12(27)
      abb12(15)=abb12(23)*abb12(15)
      abb12(23)=1.0_ki/2.0_ki*abb12(25)
      abb12(25)=spak2l4*abb12(4)*abb12(23)
      abb12(15)=abb12(15)+abb12(25)
      abb12(15)=spbk2k1*spak2l3*abb12(15)
      abb12(13)=-spbl5k1*abb12(13)
      abb12(25)=-spbl3k1*abb12(26)
      abb12(13)=abb12(13)+abb12(25)
      abb12(13)=abb12(5)*abb12(13)
      abb12(23)=abb12(23)*abb12(29)
      abb12(9)=1.0_ki/2.0_ki*abb12(9)
      R2d12=abb12(28)
      rat2 = rat2 + R2d12
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='12' value='", &
          & R2d12, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd12h8
