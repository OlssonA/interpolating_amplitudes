module     p2_gg_httbar_abbrevd83h0
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh0
   implicit none
   private
   complex(ki), dimension(49), public :: abb83
   complex(ki), public :: R2d83
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
      abb83(1)=1.0_ki/(-mT**2+es34)
      abb83(2)=sqrt(mT**2)
      abb83(3)=NC**(-1)
      abb83(4)=spbl4k2**(-1)
      abb83(5)=spbl5k2**(-1)
      abb83(6)=spak2l3**(-1)
      abb83(7)=spbl3k2**(-1)
      abb83(8)=mT*abb83(2)
      abb83(9)=i_*TR*e*gHT*abb83(1)*gs**4
      abb83(10)=abb83(8)*abb83(9)
      abb83(11)=c1*abb83(3)
      abb83(11)=abb83(11)-c3
      abb83(12)=-abb83(10)*abb83(11)
      abb83(13)=spbl3k2*abb83(4)
      abb83(14)=abb83(13)*spbe2e1
      abb83(15)=abb83(12)*abb83(14)
      abb83(16)=abb83(15)*spae2l3
      abb83(17)=abb83(9)*abb83(2)**2
      abb83(10)=abb83(17)+abb83(10)
      abb83(10)=-abb83(10)*abb83(11)
      abb83(18)=abb83(10)*spbe2e1
      abb83(19)=abb83(18)*spae2l4
      abb83(16)=abb83(16)+abb83(19)
      abb83(19)=abb83(16)*spak2l5
      abb83(18)=abb83(18)*spae2l5
      abb83(20)=abb83(18)*spak2l4
      abb83(21)=abb83(2)**3
      abb83(22)=abb83(21)*abb83(9)
      abb83(23)=abb83(17)*mT
      abb83(22)=abb83(23)+abb83(22)
      abb83(23)=-mT*abb83(11)
      abb83(22)=-abb83(22)*abb83(23)
      abb83(24)=abb83(22)*abb83(5)
      abb83(25)=spae2l4*spbe2e1
      abb83(26)=abb83(24)*abb83(25)
      abb83(19)=abb83(19)-abb83(20)+abb83(26)
      abb83(20)=spbk2k1*spae1k1
      abb83(26)=-abb83(20)*abb83(19)
      abb83(27)=-abb83(9)*abb83(11)
      abb83(8)=-abb83(27)*abb83(8)**2
      abb83(28)=spbk2e1*spae1e2
      abb83(29)=spbk2e2*abb83(5)
      abb83(30)=-abb83(8)*abb83(28)*abb83(29)*abb83(13)
      abb83(15)=abb83(15)*spae2l5
      abb83(31)=abb83(20)*abb83(15)
      abb83(30)=abb83(30)+abb83(31)
      abb83(30)=spak2l3*abb83(30)
      abb83(31)=abb83(9)*mT
      abb83(32)=abb83(21)*abb83(31)
      abb83(33)=abb83(9)*abb83(2)**4
      abb83(32)=abb83(32)+abb83(33)
      abb83(32)=abb83(11)*abb83(32)
      abb83(33)=-spae1l4*spbe2e1*abb83(32)
      abb83(21)=-abb83(27)*abb83(21)*mT
      abb83(34)=abb83(21)*abb83(14)
      abb83(35)=-spae1l3*abb83(34)
      abb83(33)=abb83(35)+abb83(33)
      abb83(33)=spae2l5*abb83(33)
      abb83(11)=-abb83(17)*abb83(11)
      abb83(17)=spak2l4*mH**2*abb83(7)*abb83(6)
      abb83(35)=abb83(17)*abb83(11)
      abb83(36)=abb83(22)*abb83(4)
      abb83(35)=abb83(35)-abb83(36)
      abb83(35)=abb83(35)*spae1e2*spbk2e2
      abb83(37)=spak1l5*spbk1e1
      abb83(38)=abb83(37)*abb83(35)
      abb83(39)=abb83(29)*spak2l4
      abb83(22)=-abb83(22)*abb83(28)*abb83(39)
      abb83(34)=spae1l5*abb83(34)
      abb83(8)=abb83(8)*abb83(5)
      abb83(14)=abb83(8)*abb83(14)
      abb83(40)=-abb83(20)*abb83(14)
      abb83(34)=abb83(34)+abb83(40)
      abb83(34)=spae2l3*abb83(34)
      abb83(40)=spal3l4*abb83(5)
      abb83(41)=-abb83(21)*abb83(40)*abb83(28)
      abb83(42)=spae1e2*abb83(11)*spal3l4
      abb83(43)=abb83(37)*abb83(42)
      abb83(41)=abb83(41)+abb83(43)
      abb83(41)=spbl3e2*abb83(41)
      abb83(25)=spae1l5*abb83(32)*abb83(25)
      abb83(11)=abb83(11)*spae1l5
      abb83(32)=abb83(11)*spal3l4
      abb83(43)=abb83(32)*spbe2e1
      abb83(44)=spae2k2*spbl3k2
      abb83(45)=abb83(44)*abb83(43)
      abb83(46)=abb83(29)*spal3l4
      abb83(21)=spbl3e1*spae1e2*abb83(21)*abb83(46)
      abb83(46)=abb83(46)*abb83(12)
      abb83(47)=abb83(46)*spae1e2
      abb83(48)=abb83(47)*spbk1e1
      abb83(49)=-spak1k2*spbl3k2*abb83(48)
      abb83(21)=abb83(49)+abb83(21)+abb83(45)+abb83(30)+abb83(41)+abb83(34)+abb&
      &83(22)+abb83(25)+abb83(38)+abb83(33)+abb83(26)
      abb83(22)=-spae1l5*abb83(16)
      abb83(25)=spbl3e2*abb83(12)*abb83(40)
      abb83(26)=abb83(28)*abb83(25)
      abb83(30)=spae1l4*abb83(18)
      abb83(33)=-spbl3e1*abb83(47)
      abb83(34)=spae1l3*abb83(15)
      abb83(22)=abb83(34)+abb83(33)+abb83(30)+abb83(26)+abb83(22)
      abb83(11)=abb83(17)*abb83(11)
      abb83(26)=abb83(36)*spae1l5
      abb83(11)=abb83(11)-abb83(26)
      abb83(26)=spbk2e2*abb83(11)
      abb83(30)=-abb83(13)*abb83(27)*mT**2
      abb83(29)=spak2l3*abb83(30)*abb83(29)
      abb83(9)=abb83(9)*abb83(2)
      abb83(9)=abb83(9)+abb83(31)
      abb83(9)=-abb83(9)*abb83(23)
      abb83(23)=abb83(39)*abb83(9)
      abb83(23)=abb83(29)+abb83(23)
      abb83(29)=abb83(20)*abb83(23)
      abb83(31)=spbl3e2*abb83(32)
      abb83(32)=-spae1k2*spbl3k2*abb83(46)
      abb83(26)=abb83(32)+abb83(31)+abb83(29)+abb83(26)
      abb83(12)=abb83(12)*abb83(13)
      abb83(29)=spae2l3*abb83(12)
      abb83(31)=abb83(10)*spae2l4
      abb83(29)=abb83(29)+abb83(31)
      abb83(31)=-spak2l5*abb83(29)
      abb83(32)=-spae2l4*abb83(24)
      abb83(10)=abb83(10)*spae2l5
      abb83(33)=spak2l4*abb83(10)
      abb83(8)=abb83(8)*abb83(13)
      abb83(13)=-spae2l3*abb83(8)
      abb83(12)=spae2l5*abb83(12)
      abb83(34)=spak2l3*abb83(12)
      abb83(13)=abb83(31)+abb83(34)+abb83(13)+abb83(32)+abb83(33)
      abb83(13)=spbk2e1*abb83(13)
      abb83(31)=abb83(27)*spal3l4
      abb83(32)=abb83(44)*abb83(31)
      abb83(33)=-abb83(37)*abb83(32)
      abb83(13)=abb83(33)+abb83(13)
      abb83(33)=spbe2k1*spae1k1
      abb83(34)=abb83(29)*abb83(33)
      abb83(36)=-spbl3e2*abb83(42)
      abb83(34)=abb83(34)+abb83(36)-abb83(35)
      abb83(29)=-2.0_ki*abb83(29)+abb83(32)
      abb83(24)=-abb83(28)*abb83(24)
      abb83(32)=abb83(9)*abb83(5)
      abb83(35)=abb83(20)*abb83(32)
      abb83(36)=-abb83(10)*abb83(33)
      abb83(10)=2.0_ki*abb83(10)
      abb83(38)=-abb83(37)*abb83(31)
      abb83(8)=-abb83(28)*abb83(8)
      abb83(28)=abb83(30)*abb83(5)
      abb83(20)=abb83(20)*abb83(28)
      abb83(30)=spak1e2*spbk1e1*abb83(46)
      abb83(39)=-2.0_ki*abb83(46)
      abb83(33)=-abb83(12)*abb83(33)
      abb83(12)=2.0_ki*abb83(12)
      abb83(11)=spbe2e1*abb83(11)
      abb83(9)=abb83(9)*abb83(4)
      abb83(17)=abb83(27)*abb83(17)
      abb83(9)=abb83(9)-abb83(17)
      abb83(17)=abb83(37)*abb83(9)
      abb83(14)=spae2l3*abb83(14)
      abb83(27)=-spak2l3*abb83(15)
      abb83(37)=abb83(25)*spbk1e1
      abb83(40)=-spak1e2*abb83(37)
      abb83(14)=abb83(40)+abb83(27)+abb83(14)+abb83(19)
      abb83(19)=2.0_ki*abb83(25)-abb83(23)
      abb83(23)=abb83(47)*spbl3k2
      abb83(16)=-spae1k1*abb83(16)
      abb83(18)=spae1k1*abb83(18)
      abb83(15)=spae1k1*abb83(15)
      abb83(25)=spae1e2*abb83(37)
      R2d83=0.0_ki
      rat2 = rat2 + R2d83
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='83' value='", &
          & R2d83, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd83h0
