module     p2_gg_httbar_abbrevd113h4
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh4
   implicit none
   private
   complex(ki), dimension(58), public :: abb113
   complex(ki), public :: R2d113
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
      abb113(1)=sqrt(mT**2)
      abb113(2)=es45**(-1)
      abb113(3)=spak2l4**(-1)
      abb113(4)=spbl5k2**(-1)
      abb113(5)=spak2l3**(-1)
      abb113(6)=spbl3k2**(-1)
      abb113(7)=spae1l5*spbl4k2
      abb113(8)=spae2k2*spbe2e1
      abb113(9)=-abb113(7)*abb113(8)
      abb113(10)=spbl4e1*spak2l5
      abb113(11)=spbk2e2*spae1e2
      abb113(12)=-abb113(10)*abb113(11)
      abb113(9)=abb113(9)+abb113(12)
      abb113(12)=abb113(5)*abb113(6)*mH**2
      abb113(13)=abb113(12)-2.0_ki
      abb113(14)=c1-c2
      abb113(14)=abb113(14)*i_*TR*e*gHT*abb113(2)*gs**4
      abb113(15)=-abb113(1)**3*abb113(14)
      abb113(9)=abb113(9)*abb113(13)*abb113(15)
      abb113(16)=mT**2*abb113(4)*abb113(3)
      abb113(14)=abb113(1)*abb113(14)
      abb113(17)=abb113(16)*abb113(14)
      abb113(18)=abb113(17)*spak2l3
      abb113(19)=abb113(18)*spbl3k2
      abb113(20)=abb113(15)*abb113(16)
      abb113(19)=abb113(19)+abb113(20)
      abb113(21)=abb113(19)*abb113(8)
      abb113(22)=spbk2k1*spae1k1
      abb113(23)=-abb113(21)*abb113(22)
      abb113(24)=abb113(19)*abb113(11)
      abb113(25)=spak1k2*spbk1e1
      abb113(26)=-abb113(24)*abb113(25)
      abb113(27)=spbl4l3*spae1l5
      abb113(28)=abb113(15)*spbe2e1
      abb113(29)=-abb113(28)*abb113(27)
      abb113(30)=spbe2e1*abb113(20)*spae1k2
      abb113(31)=-spbl3k2*abb113(30)
      abb113(29)=abb113(29)+abb113(31)
      abb113(29)=spae2l3*abb113(29)
      abb113(31)=abb113(14)*spae1k1
      abb113(32)=abb113(31)*spbl4k1
      abb113(33)=abb113(32)*spal3l5
      abb113(34)=-spbl3k2*abb113(8)*abb113(33)
      abb113(35)=abb113(14)*spbk1e1
      abb113(36)=abb113(35)*spak1l5
      abb113(37)=abb113(36)*spbl4l3
      abb113(38)=-spak2l3*abb113(11)*abb113(37)
      abb113(39)=spal3l5*spbl4e1
      abb113(16)=-spbk2e1*abb113(16)*spak2l3
      abb113(16)=-abb113(39)+abb113(16)
      abb113(40)=abb113(15)*spae1e2
      abb113(16)=spbl3e2*abb113(40)*abb113(16)
      abb113(41)=abb113(28)*spae2l5
      abb113(42)=spbl4k1*spae1k1*abb113(41)
      abb113(43)=abb113(40)*spbl4e2
      abb113(44)=spak1l5*spbk1e1*abb113(43)
      abb113(9)=abb113(29)+abb113(16)+abb113(44)+abb113(42)+abb113(26)+abb113(2&
      &3)+abb113(38)+abb113(34)+abb113(9)
      abb113(16)=abb113(17)*abb113(11)
      abb113(23)=abb113(16)*abb113(25)
      abb113(26)=abb113(17)*abb113(8)
      abb113(29)=abb113(26)*abb113(22)
      abb113(23)=abb113(23)+abb113(29)
      abb113(29)=-2.0_ki*abb113(23)
      abb113(34)=-spae2l5*spbe2e1*abb113(32)
      abb113(38)=-spbl4e2*spae1e2*abb113(36)
      abb113(23)=abb113(38)+abb113(34)+abb113(23)
      abb113(31)=abb113(31)*spbl4k1*spak2l5
      abb113(34)=-abb113(13)*abb113(31)
      abb113(38)=abb113(14)*spak2l3
      abb113(27)=-abb113(38)*abb113(27)
      abb113(27)=abb113(34)+abb113(27)
      abb113(27)=spbk2e2*abb113(27)
      abb113(34)=spae1k2*spbk2e2
      abb113(42)=-abb113(19)*abb113(34)
      abb113(44)=abb113(22)*abb113(18)
      abb113(33)=abb113(44)+abb113(33)
      abb113(44)=-spbl3e2*abb113(33)
      abb113(45)=spbl4e2*spae1l5*abb113(15)
      abb113(27)=abb113(44)+abb113(45)+abb113(42)+abb113(27)
      abb113(42)=2.0_ki*abb113(17)
      abb113(44)=-abb113(42)*abb113(34)
      abb113(34)=abb113(17)*abb113(34)
      abb113(45)=abb113(14)*spae1l5
      abb113(46)=-spbl4e2*abb113(45)
      abb113(34)=abb113(34)+abb113(46)
      abb113(35)=abb113(35)*spak1l5*spbl4k2
      abb113(13)=-abb113(13)*abb113(35)
      abb113(46)=abb113(14)*spbl3k2
      abb113(39)=-abb113(46)*abb113(39)
      abb113(13)=abb113(13)+abb113(39)
      abb113(13)=spae2k2*abb113(13)
      abb113(39)=spbk2e1*spae2k2
      abb113(19)=-abb113(19)*abb113(39)
      abb113(47)=abb113(17)*spbl3k2
      abb113(48)=abb113(25)*abb113(47)
      abb113(37)=abb113(48)+abb113(37)
      abb113(48)=-spae2l3*abb113(37)
      abb113(15)=spae2l5*spbl4e1*abb113(15)
      abb113(13)=abb113(48)+abb113(15)+abb113(19)+abb113(13)
      abb113(15)=-abb113(42)*abb113(39)
      abb113(17)=abb113(17)*abb113(39)
      abb113(19)=abb113(14)*spbl4e1
      abb113(39)=-spae2l5*abb113(19)
      abb113(17)=abb113(17)+abb113(39)
      abb113(39)=2.0_ki*spbl4e1*abb113(40)
      abb113(40)=2.0_ki*abb113(32)
      abb113(38)=spbl4l3*abb113(11)*abb113(38)
      abb113(38)=abb113(38)-abb113(43)
      abb113(43)=abb113(14)*spae1e2
      abb113(48)=spbl4e2*abb113(43)
      abb113(49)=abb113(12)*abb113(14)
      abb113(50)=2.0_ki*abb113(14)
      abb113(51)=abb113(49)-abb113(50)
      abb113(52)=spbl4k2*spae2k2*abb113(51)
      abb113(53)=spae2l3*spbl4l3*abb113(14)
      abb113(52)=abb113(52)+abb113(53)
      abb113(28)=2.0_ki*spae1l5*abb113(28)
      abb113(53)=2.0_ki*abb113(36)
      abb113(46)=spal3l5*abb113(8)*abb113(46)
      abb113(41)=abb113(46)-abb113(41)
      abb113(46)=abb113(14)*spbe2e1
      abb113(54)=spae2l5*abb113(46)
      abb113(51)=spak2l5*spbk2e2*abb113(51)
      abb113(55)=spbl3e2*spal3l5*abb113(14)
      abb113(51)=abb113(51)+abb113(55)
      abb113(33)=-spbe2e1*abb113(33)
      abb113(55)=-spbk2e1*abb113(18)
      abb113(56)=-spal3l5*abb113(19)
      abb113(55)=abb113(56)+abb113(55)
      abb113(56)=spal3l5*abb113(46)
      abb113(37)=-spae1e2*abb113(37)
      abb113(57)=-spae1k2*abb113(47)
      abb113(58)=-spbl4l3*abb113(45)
      abb113(57)=abb113(58)+abb113(57)
      abb113(58)=spbl4l3*abb113(43)
      abb113(12)=abb113(12)-1.0_ki
      abb113(31)=-spbe2e1*abb113(12)*abb113(31)
      abb113(30)=abb113(31)-2.0_ki*abb113(30)
      abb113(31)=abb113(49)-abb113(14)
      abb113(10)=-abb113(31)*abb113(10)
      abb113(25)=-abb113(42)*abb113(25)
      abb113(10)=abb113(10)+abb113(25)
      abb113(25)=spbe2e1*abb113(49)
      abb113(25)=-abb113(46)+abb113(25)
      abb113(25)=spak2l5*abb113(25)
      abb113(12)=-abb113(12)*abb113(35)
      abb113(20)=spbk2e1*abb113(20)
      abb113(12)=abb113(12)-2.0_ki*abb113(20)
      abb113(12)=spae1e2*abb113(12)
      abb113(7)=-abb113(31)*abb113(7)
      abb113(20)=-abb113(42)*abb113(22)
      abb113(7)=abb113(7)+abb113(20)
      abb113(20)=spae1e2*abb113(49)
      abb113(20)=-abb113(43)+abb113(20)
      abb113(20)=spbl4k2*abb113(20)
      abb113(22)=abb113(8)*abb113(42)
      abb113(31)=spbl3e2*abb113(18)
      abb113(18)=spbe2e1*abb113(18)
      abb113(35)=abb113(11)*abb113(42)
      abb113(43)=spae2l3*abb113(47)
      abb113(46)=spae1e2*abb113(47)
      abb113(32)=abb113(8)*abb113(32)
      abb113(19)=spae2k2*abb113(19)
      abb113(8)=-abb113(14)*abb113(8)
      abb113(36)=abb113(11)*abb113(36)
      abb113(45)=spbk2e2*abb113(45)
      abb113(11)=-abb113(14)*abb113(11)
      R2d113=0.0_ki
      rat2 = rat2 + R2d113
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='113' value='", &
          & R2d113, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd113h4
