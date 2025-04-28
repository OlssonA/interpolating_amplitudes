module     p2_gg_httbar_abbrevd68h0
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh0
   implicit none
   private
   complex(ki), dimension(49), public :: abb68
   complex(ki), public :: R2d68
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
      abb68(1)=sqrt(mT**2)
      abb68(2)=NC**(-1)
      abb68(3)=es12**(-1)
      abb68(4)=spbl5k2**(-1)
      abb68(5)=spbl4k2**(-1)
      abb68(6)=spak2l3**(-1)
      abb68(7)=spbl3k2**(-1)
      abb68(8)=spak2l4**(-1)
      abb68(9)=spak2l5**(-1)
      abb68(10)=spal4l5*spbl5k1
      abb68(11)=spal3l4*spbl3k1
      abb68(10)=abb68(10)+abb68(11)
      abb68(10)=abb68(10)*spak1l5
      abb68(12)=spal4l5*spbl5k2
      abb68(13)=spbl3k2*spal3l4
      abb68(12)=abb68(12)+abb68(13)
      abb68(12)=abb68(12)*spak2l5
      abb68(14)=spal4l5*spbl4k2
      abb68(15)=spbl3k2*spal3l5
      abb68(14)=abb68(14)-abb68(15)
      abb68(14)=abb68(14)*spak2l4
      abb68(16)=spak1l4*spak2l5
      abb68(17)=mH**2*abb68(6)*abb68(7)
      abb68(18)=abb68(16)*abb68(17)
      abb68(19)=abb68(17)*spak2l4
      abb68(20)=abb68(19)*spak1l5
      abb68(18)=abb68(18)-abb68(20)
      abb68(18)=abb68(18)*spbk2k1
      abb68(21)=spal4l5*spbl4k1
      abb68(22)=spbl3k1*spal3l5
      abb68(23)=abb68(21)-abb68(22)
      abb68(23)=abb68(23)*spak1l4
      abb68(10)=-abb68(10)+abb68(18)-abb68(23)+abb68(12)+abb68(14)
      abb68(12)=c2-c1
      abb68(10)=abb68(12)*abb68(10)
      abb68(14)=abb68(5)*spak1l5
      abb68(18)=abb68(4)*spak1l4
      abb68(23)=abb68(14)+abb68(18)
      abb68(24)=-spbk2k1*abb68(12)
      abb68(25)=abb68(1)*mT
      abb68(23)=-abb68(25)*abb68(23)*abb68(24)
      abb68(26)=mT**2
      abb68(27)=abb68(24)*abb68(26)
      abb68(28)=abb68(9)*spak1l5*spak2l4
      abb68(28)=abb68(28)-spak1l4
      abb68(28)=abb68(28)*abb68(4)
      abb68(29)=abb68(16)*abb68(8)
      abb68(29)=abb68(29)-spak1l5
      abb68(29)=abb68(29)*abb68(5)
      abb68(28)=abb68(28)-abb68(29)
      abb68(28)=abb68(28)*abb68(27)
      abb68(10)=abb68(28)+abb68(23)-abb68(10)
      abb68(23)=abb68(1)*abb68(10)
      abb68(18)=abb68(18)*spbl4k2
      abb68(28)=abb68(18)*abb68(19)
      abb68(28)=abb68(28)+abb68(20)
      abb68(28)=abb68(28)*spbk2k1
      abb68(29)=abb68(5)*spbl3k2
      abb68(30)=abb68(29)*spak1l3
      abb68(31)=abb68(30)*abb68(21)
      abb68(32)=abb68(11)*spak1l4
      abb68(33)=abb68(13)*spak2l4
      abb68(32)=abb68(32)-abb68(33)
      abb68(33)=abb68(4)*spbl4k2
      abb68(34)=abb68(32)*abb68(33)
      abb68(35)=abb68(11)*spak1l5
      abb68(36)=abb68(13)*spak2l5
      abb68(35)=abb68(35)-abb68(36)
      abb68(36)=spak2l3*spbl3k2*spal4l5
      abb68(28)=-abb68(36)+abb68(28)+abb68(35)+abb68(31)+abb68(34)
      abb68(28)=abb68(28)*abb68(12)
      abb68(31)=abb68(30)*abb68(4)
      abb68(34)=spak2l5*abb68(8)
      abb68(36)=abb68(34)*abb68(5)**2
      abb68(37)=abb68(36)*spbl3k2*spak1l3
      abb68(31)=abb68(31)+abb68(37)
      abb68(31)=-abb68(31)*abb68(27)
      abb68(28)=abb68(31)+abb68(28)
      abb68(28)=mT*abb68(28)
      abb68(23)=abb68(28)+abb68(23)
      abb68(28)=e*gs**4*abb68(2)*gHT*spbe2e1*spae1e2*TR*i_
      abb68(31)=abb68(28)*abb68(3)
      abb68(37)=1.0_ki/2.0_ki*abb68(31)
      abb68(38)=abb68(37)*abb68(1)
      abb68(23)=abb68(23)*abb68(38)
      abb68(39)=abb68(12)*abb68(31)
      abb68(20)=abb68(20)*spbk2k1
      abb68(20)=abb68(20)+abb68(35)
      abb68(20)=abb68(20)*abb68(39)
      abb68(35)=abb68(17)*spbk2k1
      abb68(16)=abb68(16)*abb68(35)
      abb68(40)=abb68(22)*spak1l4
      abb68(41)=abb68(15)*spak2l4
      abb68(16)=-abb68(41)+abb68(40)+abb68(16)
      abb68(16)=-abb68(16)*abb68(39)
      abb68(39)=abb68(25)*abb68(31)
      abb68(40)=abb68(12)*abb68(39)
      abb68(14)=abb68(14)*abb68(35)
      abb68(13)=abb68(13)*abb68(4)
      abb68(13)=-abb68(14)-abb68(13)+2.0_ki*spal4l5
      abb68(13)=abb68(13)*abb68(40)
      abb68(10)=abb68(10)*abb68(37)
      abb68(14)=abb68(12)*abb68(37)
      abb68(41)=spak1l4*abb68(19)*spbk2k1
      abb68(32)=abb68(41)+abb68(32)
      abb68(32)=abb68(32)*abb68(14)
      abb68(41)=abb68(5)*abb68(8)
      abb68(42)=abb68(41)*abb68(11)
      abb68(11)=abb68(11)*abb68(4)
      abb68(43)=abb68(11)*abb68(9)
      abb68(42)=abb68(43)+abb68(42)
      abb68(26)=abb68(12)*abb68(26)
      abb68(42)=abb68(26)*spak1k2*abb68(42)
      abb68(43)=abb68(12)*mT
      abb68(29)=abb68(29)*spak2l3*abb68(43)
      abb68(44)=3.0_ki*abb68(1)
      abb68(45)=abb68(12)*spak2l4
      abb68(46)=abb68(45)*abb68(44)
      abb68(29)=abb68(29)+abb68(46)
      abb68(29)=abb68(1)*abb68(29)
      abb68(29)=abb68(42)+abb68(29)
      abb68(29)=abb68(3)*abb68(29)
      abb68(19)=abb68(19)*abb68(4)
      abb68(42)=abb68(19)*abb68(9)
      abb68(46)=abb68(17)*abb68(5)
      abb68(42)=abb68(42)+abb68(46)
      abb68(42)=abb68(42)*abb68(26)
      abb68(29)=abb68(42)+abb68(29)
      abb68(42)=1.0_ki/2.0_ki*abb68(28)
      abb68(29)=abb68(29)*abb68(42)
      abb68(42)=-abb68(5)*abb68(12)
      abb68(47)=abb68(42)*abb68(39)
      abb68(48)=2.0_ki*abb68(47)
      abb68(45)=abb68(45)*abb68(37)
      abb68(30)=-abb68(30)*abb68(43)
      abb68(49)=abb68(12)*spak1l4
      abb68(44)=-abb68(49)*abb68(44)
      abb68(30)=abb68(30)+abb68(44)
      abb68(30)=abb68(30)*abb68(38)
      abb68(38)=-abb68(49)*abb68(37)
      abb68(35)=abb68(35)*spak1l5
      abb68(15)=abb68(35)-abb68(15)
      abb68(15)=abb68(15)*spak2l5
      abb68(35)=abb68(22)*spak1l5
      abb68(15)=abb68(15)+abb68(35)
      abb68(14)=-abb68(15)*abb68(14)
      abb68(15)=abb68(37)*abb68(25)
      abb68(24)=abb68(15)*abb68(4)*spak1l3*abb68(24)
      abb68(34)=abb68(46)*abb68(34)
      abb68(17)=abb68(17)*abb68(4)
      abb68(34)=abb68(34)+abb68(17)
      abb68(34)=-abb68(34)*abb68(26)
      abb68(17)=-abb68(25)*abb68(17)*abb68(12)
      abb68(17)=abb68(34)+abb68(17)
      abb68(25)=abb68(4)*abb68(9)
      abb68(25)=abb68(41)+abb68(25)
      abb68(22)=-abb68(25)*abb68(22)*spak1k2*abb68(26)
      abb68(25)=spak2l4*abb68(33)
      abb68(25)=abb68(25)+spak2l5
      abb68(25)=-abb68(25)*abb68(43)
      abb68(26)=3.0_ki/2.0_ki*abb68(1)
      abb68(33)=abb68(12)*spak2l5
      abb68(34)=-abb68(33)*abb68(26)
      abb68(25)=abb68(25)+abb68(34)
      abb68(25)=abb68(1)*abb68(25)
      abb68(22)=1.0_ki/2.0_ki*abb68(22)+abb68(25)
      abb68(22)=abb68(3)*abb68(22)
      abb68(17)=1.0_ki/2.0_ki*abb68(17)+abb68(22)
      abb68(17)=abb68(17)*abb68(28)
      abb68(22)=abb68(4)*abb68(40)
      abb68(25)=-2.0_ki*abb68(22)
      abb68(28)=-abb68(33)*abb68(37)
      abb68(18)=abb68(18)+spak1l5
      abb68(18)=abb68(18)*abb68(43)
      abb68(33)=abb68(12)*spak1l5
      abb68(26)=abb68(33)*abb68(26)
      abb68(18)=abb68(18)+abb68(26)
      abb68(18)=abb68(1)*abb68(18)*abb68(31)
      abb68(26)=abb68(33)*abb68(37)
      abb68(31)=spak2l5*spbl3k2
      abb68(33)=spak1l5*spbl3k1
      abb68(31)=abb68(31)-abb68(33)
      abb68(15)=-abb68(15)*abb68(31)*abb68(42)
      abb68(31)=abb68(4)*abb68(5)
      abb68(31)=abb68(31)+abb68(36)
      abb68(27)=abb68(31)*abb68(27)
      abb68(19)=abb68(19)*spbk2k1
      abb68(11)=abb68(19)+abb68(11)
      abb68(19)=abb68(21)*abb68(5)
      abb68(11)=-abb68(19)+1.0_ki/2.0_ki*abb68(11)
      abb68(11)=abb68(11)*abb68(12)
      abb68(11)=abb68(27)+abb68(11)
      abb68(11)=abb68(11)*abb68(39)
      R2d68=0.0_ki
      rat2 = rat2 + R2d68
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='68' value='", &
          & R2d68, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd68h0
