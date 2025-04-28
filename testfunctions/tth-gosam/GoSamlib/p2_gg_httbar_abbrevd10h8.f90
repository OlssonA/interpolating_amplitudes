module     p2_gg_httbar_abbrevd10h8
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh8
   implicit none
   private
   complex(ki), dimension(33), public :: abb10
   complex(ki), public :: R2d10
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
      abb10(1)=1.0_ki/(-mT**2+es34)
      abb10(2)=NC**(-1)
      abb10(3)=es12**(-1)
      abb10(4)=spbl4k2**(-1)
      abb10(5)=spak2l5**(-1)
      abb10(6)=spak2l3**(-1)
      abb10(7)=spbl3k2**(-1)
      abb10(8)=sqrt(mT**2)
      abb10(9)=mT**3
      abb10(10)=spbe2e1*spae1e2*gs**4*i_*TR*e*gHT*abb10(2)*abb10(1)
      abb10(11)=abb10(9)*abb10(10)
      abb10(12)=mT**2
      abb10(13)=abb10(12)*abb10(10)
      abb10(14)=abb10(13)*abb10(8)
      abb10(11)=abb10(14)+abb10(11)
      abb10(14)=c1-c2
      abb10(15)=abb10(4)*abb10(5)
      abb10(11)=-abb10(15)*abb10(11)*abb10(14)
      abb10(16)=abb10(14)*abb10(10)*mT
      abb10(17)=spak2l4*abb10(5)
      abb10(18)=mH**2*abb10(7)*abb10(6)
      abb10(19)=abb10(17)*abb10(18)
      abb10(20)=abb10(16)*abb10(19)
      abb10(21)=spbl5k2*spak2l4
      abb10(22)=spak1l4*spbl5k1
      abb10(21)=abb10(21)-abb10(22)
      abb10(10)=abb10(10)*abb10(3)
      abb10(22)=abb10(8)*abb10(10)
      abb10(23)=mT*abb10(10)
      abb10(24)=abb10(22)+abb10(23)
      abb10(24)=-abb10(24)*abb10(14)
      abb10(21)=abb10(21)*abb10(24)
      abb10(20)=abb10(21)+abb10(11)-abb10(20)
      abb10(21)=spak2l3*spbl5k2
      abb10(25)=spak1l3*spbl5k1
      abb10(21)=abb10(21)-abb10(25)
      abb10(25)=-abb10(23)*abb10(14)
      abb10(26)=spbl3k2*abb10(4)
      abb10(27)=abb10(25)*abb10(26)
      abb10(28)=abb10(27)*abb10(21)
      abb10(29)=spal3l4*abb10(5)
      abb10(30)=abb10(29)*abb10(25)
      abb10(31)=spbl3k1*spak1k2
      abb10(32)=abb10(30)*abb10(31)
      abb10(28)=abb10(32)+abb10(28)+abb10(20)
      abb10(28)=1.0_ki/4.0_ki*abb10(28)
      abb10(32)=abb10(8)**2
      abb10(20)=abb10(32)*abb10(20)
      abb10(21)=-abb10(26)*abb10(21)
      abb10(31)=-abb10(29)*abb10(31)
      abb10(21)=abb10(31)+abb10(21)
      abb10(21)=-abb10(21)*abb10(32)*abb10(25)
      abb10(20)=abb10(20)+abb10(21)
      abb10(20)=1.0_ki/2.0_ki*abb10(20)
      abb10(13)=abb10(13)*abb10(3)
      abb10(21)=abb10(23)*abb10(8)
      abb10(13)=abb10(21)+abb10(13)
      abb10(13)=-abb10(14)*abb10(13)*abb10(8)
      abb10(21)=abb10(13)*abb10(17)
      abb10(12)=abb10(22)*abb10(12)
      abb10(23)=-abb10(12)*abb10(14)
      abb10(26)=abb10(23)*abb10(26)
      abb10(31)=spak2l3*abb10(5)*abb10(26)
      abb10(21)=abb10(21)+abb10(31)
      abb10(17)=abb10(16)*abb10(17)
      abb10(22)=-abb10(22)*abb10(14)
      abb10(31)=abb10(22)*spak2l4
      abb10(32)=2.0_ki*spbl5k2
      abb10(33)=-abb10(31)*abb10(32)
      abb10(17)=abb10(17)+abb10(33)
      abb10(17)=abb10(17)*abb10(18)
      abb10(33)=abb10(13)*abb10(4)
      abb10(32)=-abb10(33)*abb10(32)
      abb10(11)=abb10(17)+abb10(32)-abb10(11)+2.0_ki*abb10(21)
      abb10(9)=abb10(9)*abb10(10)
      abb10(9)=abb10(9)+abb10(12)
      abb10(9)=-abb10(15)*abb10(9)*abb10(14)
      abb10(10)=abb10(19)*abb10(25)
      abb10(9)=abb10(9)+abb10(10)
      abb10(10)=-2.0_ki*abb10(9)
      abb10(12)=1.0_ki/2.0_ki*abb10(24)
      abb10(14)=spak1k2*spbl5k1
      abb10(17)=abb10(12)*abb10(14)
      abb10(19)=abb10(5)*abb10(13)
      abb10(21)=spbl5k2*abb10(12)
      abb10(19)=abb10(19)+abb10(21)
      abb10(19)=spak1k2*abb10(19)
      abb10(21)=1.0_ki/2.0_ki*abb10(27)
      abb10(14)=abb10(21)*abb10(14)
      abb10(15)=abb10(23)*abb10(15)
      abb10(23)=1.0_ki/2.0_ki*abb10(25)
      abb10(24)=spbl5k2*abb10(4)*abb10(23)
      abb10(15)=abb10(15)+abb10(24)
      abb10(15)=spak1k2*spbl3k2*abb10(15)
      abb10(24)=abb10(22)*spal3l4
      abb10(25)=spbl3k2*abb10(24)
      abb10(16)=abb10(5)*abb10(16)
      abb10(22)=-spbl5k2*abb10(22)
      abb10(16)=1.0_ki/2.0_ki*abb10(16)+abb10(22)
      abb10(16)=spal3l4*abb10(16)
      abb10(13)=-spak1l4*abb10(13)
      abb10(22)=-spak1l3*abb10(26)
      abb10(13)=abb10(13)+abb10(22)
      abb10(13)=abb10(5)*abb10(13)
      abb10(18)=abb10(31)*abb10(18)
      abb10(18)=abb10(18)+abb10(33)
      abb10(22)=-spbk2k1*abb10(18)
      abb10(26)=-spbl3k1*abb10(24)
      abb10(22)=abb10(26)+abb10(22)
      abb10(24)=spbl5k1*abb10(24)
      abb10(23)=abb10(23)*abb10(29)
      abb10(18)=spbl5k1*abb10(18)
      abb10(9)=1.0_ki/2.0_ki*abb10(9)
      R2d10=abb10(28)
      rat2 = rat2 + R2d10
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='10' value='", &
          & R2d10, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd10h8
